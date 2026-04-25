"""AWQ preprocessor.

For each fusable group of linears that share an input, search a
per-input-channel scale s that minimizes int4 reconstruction error of the
linears' outputs, multiply s into the linears' weights, and absorb 1/s
into the preceding RMSNorm or up_proj for the down_proj group.

No weight-clipping / o_proj scaling / asymmetric int4 as in original AWQ.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

_INT4_MAX = 7
_INT4_MIN = -8


@torch.no_grad()
def _rtn_roundtrip(w: torch.Tensor, group_size: int) -> torch.Tensor:
    out_f, in_f = w.shape
    assert in_f % group_size == 0, (in_f, group_size)
    w_g = w.reshape(out_f, in_f // group_size, group_size).float()
    scale = w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / _INT4_MAX
    q = (w_g / scale).round().clamp(_INT4_MIN, _INT4_MAX)
    return (q * scale).reshape(out_f, in_f).to(w.dtype)


@torch.no_grad()
def _search_scale(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    group_size: int,
    n_grid: int = 10,
) -> torch.Tensor:
    x = x.float()
    x_mean = x.abs().mean(dim=0).clamp(min=1e-4)
    y_refs = [x @ w.float().T for w in weights]

    best_s = torch.ones_like(x_mean)
    best_loss = float("inf")
    for alpha_idx in range(n_grid):
        alpha = alpha_idx / (n_grid - 1)
        s = x_mean.pow(alpha)
        s = s / (s.max() * s.min()).sqrt()

        loss = 0.0
        x_scaled = x / s
        for w, y_ref in zip(weights, y_refs):
            w_scaled = w.float() * s
            w_hat = _rtn_roundtrip(w_scaled, group_size).float()
            y_hat = x_scaled @ w_hat.T
            loss += (y_ref - y_hat).pow(2).mean().item()

        if loss < best_loss:
            best_loss, best_s = loss, s
    return best_s


@torch.no_grad()
def _apply_scale_qkv_gateup(
    norm: nn.Module,
    linears: Iterable[nn.Linear],
    s: torch.Tensor,
) -> None:
    s = s.to(norm.weight.dtype)
    norm.weight.data.div_(s)
    for lin in linears:
        lin.weight.data.mul_(s.to(lin.weight.dtype))


@torch.no_grad()
def _apply_scale_down(
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    s: torch.Tensor,
) -> None:
    up_proj.weight.data.div_(s.to(up_proj.weight.dtype).unsqueeze(1))
    down_proj.weight.data.mul_(s.to(down_proj.weight.dtype).unsqueeze(0))


@torch.no_grad()
def _collect_inputs(model, decoder_layers, calib_ids, device):
    collected: list[dict[str, list[torch.Tensor]]] = [
        {"qkv": [], "gateup": [], "down": []} for _ in decoder_layers
    ]
    handles = []
    for i, layer in enumerate(decoder_layers):
        attn, mlp = layer.self_attn, layer.mlp

        def qkv_hook(_mod, inp, _out, i=i):
            collected[i]["qkv"].append(inp[0].detach().reshape(-1, inp[0].shape[-1]))

        def gateup_hook(_mod, inp, _out, i=i):
            collected[i]["gateup"].append(inp[0].detach().reshape(-1, inp[0].shape[-1]))

        def down_hook(_mod, inp, _out, i=i):
            collected[i]["down"].append(inp[0].detach().reshape(-1, inp[0].shape[-1]))

        handles.append(attn.q_proj.register_forward_hook(qkv_hook))
        handles.append(mlp.gate_proj.register_forward_hook(gateup_hook))
        handles.append(mlp.down_proj.register_forward_hook(down_hook))

    try:
        for ids in calib_ids:
            model(ids.to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return [{k: torch.cat(v, dim=0) for k, v in d.items()} for d in collected]


@torch.no_grad()
def awq_preprocess(
    model,
    calib_ids: list[torch.Tensor],
    group_size: int = 128,
    n_grid: int = 10,
):
    device = next(model.parameters()).device
    decoder_layers = model.model.layers

    per_layer = _collect_inputs(model, decoder_layers, calib_ids, device)

    for i, layer in enumerate(decoder_layers):
        attn, mlp = layer.self_attn, layer.mlp
        acts = per_layer[i]

        s_qkv = _search_scale(
            acts["qkv"],
            [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight],
            group_size, n_grid,
        )
        _apply_scale_qkv_gateup(
            layer.input_layernorm, [attn.q_proj, attn.k_proj, attn.v_proj], s_qkv,
        )

        s_gu = _search_scale(
            acts["gateup"],
            [mlp.gate_proj.weight, mlp.up_proj.weight],
            group_size, n_grid,
        )
        _apply_scale_qkv_gateup(
            layer.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj], s_gu,
        )

        s_d = _search_scale(
            acts["down"], [mlp.down_proj.weight], group_size, n_grid,
        )
        _apply_scale_down(mlp.up_proj, mlp.down_proj, s_d)

        per_layer[i] = None

    return model


def make_wikitext_calib(tokenizer, n_samples: int = 16, seq_len: int = 512):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = []
    stride = ids.numel() // n_samples
    for i in range(n_samples):
        start = i * stride
        chunk = ids[start : start + seq_len]
        if chunk.numel() == seq_len:
            chunks.append(chunk.unsqueeze(0))
    return chunks
