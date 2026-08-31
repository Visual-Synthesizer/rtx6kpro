"""rp-native tiny-M (decode) MoE for w4a8_mx — Triton v3.

Reads the N256/K128 repacked weights + sfb grids directly (verified mappings),
BF16 activations (no input quant), fp32 accumulation, SiLU-gated FC1 -> FC2.

Key structure: one rp (nt,kt) tile = 4096 contiguous int32 words whose flat
index decomposes as  flat = k32<<10 | n8c<<7 | r8<<4 | cgrp<<2 | n8i.
So we load the tile with a plain contiguous arange (vectorizable) and
tl.reshape to (k32=4, n8c=8, r8=8, cgrp=4, n8i=4); logical coords follow from
the axes:  row(in tile) = n8c*32 + n8i*8 + r8,  k = kt*128 + (k32*4+cgrp)*8 + j.
Scale col = kt*4 + k32 (whole word shares one 32-group).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_val(nib):
    # e2m1 nibble -> fp32 by direct bit assembly (no SFU):
    #   e>0: (1.m) * 2^(e-1)  -> fp32 exp field = e + 126
    #   e=0: m * 0.5
    s = (nib >> 3) & 1
    e = (nib >> 1) & 3
    m = nib & 1
    bits = tl.where(e > 0, ((e + 126) << 23) | (m << 22), m * (126 << 23))
    bits = bits | (s << 31)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _tiny_fc1_kernel(
    x_ptr, ids_ptr, w13_ptr, sfb13_ptr, inter_ptr,
    TOPK: tl.constexpr,
    K: tl.constexpr, N2: tl.constexpr,
    W13_STRIDE: tl.constexpr, SFB13_STRIDE: tl.constexpr,
    ROT: tl.constexpr, KT_TILES: tl.constexpr, KT_PER_PROG: tl.constexpr,
):
    # grid: (M*topk, N2//256, KT_TILES//KT_PER_PROG)
    pid_rt = tl.program_id(0)
    nt = tl.program_id(1)
    pid_k = tl.program_id(2)

    eid = tl.load(ids_ptr + pid_rt).to(tl.int64)
    tok = pid_rt // TOPK
    w_base = eid * W13_STRIDE
    s_base = eid * SFB13_STRIDE

    k32 = tl.arange(0, 4)[:, None, None, None, None]
    n8c = tl.arange(0, 8)[None, :, None, None, None]
    r8 = tl.arange(0, 8)[None, None, :, None, None]
    cgrp = tl.arange(0, 4)[None, None, None, :, None]
    v = tl.arange(0, 4)[None, None, None, None, :]

    acc = tl.zeros((4, 8, 8, 4, 4), dtype=tl.float32)
    for kt_i in tl.range(0, KT_PER_PROG):
        kt = pid_k * KT_PER_PROG + kt_i
        tile_base = w_base + (nt * KT_TILES + kt).to(tl.int64) * 4096
        flat = tl.arange(0, 4096)
        words = tl.reshape(tl.load(w13_ptr + tile_base + flat), (4, 8, 8, 4, 4))
        # per-word 32-group scale: col = kt*4 + k32 -> kb=k32, col-tile=kt
        sfb_off = k32 | (r8 << 2) | ((n8c * 4 + v) << 5) | ((nt * KT_TILES + kt) << 10)
        wscale = (tl.load(sfb13_ptr + s_base + sfb_off).to(tl.int32) << 23).to(
            tl.float32, bitcast=True
        )
        a128 = tl.load(x_ptr + tok * K + kt * 128 + tl.arange(0, 128)).to(tl.float32)
        aw = tl.reshape(a128, (16, 8))  # (k32*4+cgrp, j)
        jj = tl.arange(0, 8)[None, :]
        part = tl.zeros((4, 8, 8, 4, 4), dtype=tl.float32)
        for j in tl.static_range(8):
            wv = _fp4_val((words >> (4 * j)) & 0xF)
            aj = tl.sum(tl.where(jj == j, aw, 0.0), axis=1)          # (16,)
            a = tl.reshape(aj, (4, 1, 1, 4, 1))                       # (k32,-,-,cgrp,-)
            part += wv * a
        acc += part * wscale
    row_part = tl.sum(tl.sum(acc, axis=3), axis=0)  # -> (n8c=8, r8=8, n8i=4)
    n8c_r = tl.arange(0, 8)[:, None, None]
    r8_r = tl.arange(0, 8)[None, :, None]
    v_r = tl.arange(0, 4)[None, None, :]
    p_full = nt * 256 + n8c_r * 32 + v_r * 8 + r8_r
    r_log = (p_full + ROT) % N2
    tl.atomic_add(inter_ptr + pid_rt.to(tl.int64) * N2 + r_log, row_part)


@triton.jit
def _tiny_fc2_kernel(
    inter_ptr, ids_ptr, tw_ptr, w2_ptr, sfb2_ptr, out_ptr,
    TOPK: tl.constexpr,
    N: tl.constexpr, K_OUT: tl.constexpr,
    W2_STRIDE: tl.constexpr, SFB2_STRIDE: tl.constexpr,
    KT_TILES: tl.constexpr, KT_PER_PROG: tl.constexpr,
):
    # grid: (M*topk, K_OUT//256, KT_TILES//KT_PER_PROG)
    pid_rt = tl.program_id(0)
    nt = tl.program_id(1)
    pid_k = tl.program_id(2)

    eid = tl.load(ids_ptr + pid_rt).to(tl.int64)
    tok = pid_rt // TOPK
    rw = tl.load(tw_ptr + pid_rt)
    w_base = eid * W2_STRIDE
    s_base = eid * SFB2_STRIDE

    k32 = tl.arange(0, 4)[:, None, None, None, None]
    n8c = tl.arange(0, 8)[None, :, None, None, None]
    r8 = tl.arange(0, 8)[None, None, :, None, None]
    cgrp = tl.arange(0, 4)[None, None, None, :, None]
    v = tl.arange(0, 4)[None, None, None, None, :]

    ibase = pid_rt.to(tl.int64) * (2 * N)
    acc = tl.zeros((4, 8, 8, 4, 4), dtype=tl.float32)
    for kt_i in tl.range(0, KT_PER_PROG):
        kt = pid_k * KT_PER_PROG + kt_i
        tile_base = w_base + (nt * KT_TILES + kt).to(tl.int64) * 4096
        flat = tl.arange(0, 4096)
        words = tl.reshape(tl.load(w2_ptr + tile_base + flat), (4, 8, 8, 4, 4))
        sfb_off = k32 | (r8 << 2) | ((n8c * 4 + v) << 5) | ((nt * KT_TILES + kt) << 10)
        wscale = (tl.load(sfb2_ptr + s_base + sfb_off).to(tl.int32) << 23).to(
            tl.float32, bitcast=True
        )
        g128 = tl.load(inter_ptr + ibase + kt * 128 + tl.arange(0, 128))
        u128 = tl.load(inter_ptr + ibase + N + kt * 128 + tl.arange(0, 128))
        act128 = (g128 / (1.0 + tl.exp(-g128))) * u128               # silu(gate)*up
        aw = tl.reshape(act128, (16, 8))
        jj = tl.arange(0, 8)[None, :]
        part = tl.zeros((4, 8, 8, 4, 4), dtype=tl.float32)
        for j in tl.static_range(8):
            wv = _fp4_val((words >> (4 * j)) & 0xF)
            aj = tl.sum(tl.where(jj == j, aw, 0.0), axis=1)
            a = tl.reshape(aj, (4, 1, 1, 4, 1))
            part += wv * a
        acc += part * wscale
    row_part = tl.sum(tl.sum(acc, axis=3), axis=0)  # (8, 8, 4)
    n8c_r = tl.arange(0, 8)[:, None, None]
    r8_r = tl.arange(0, 8)[None, :, None]
    v_r = tl.arange(0, 4)[None, None, :]
    p_full = nt * 256 + n8c_r * 32 + v_r * 8 + r8_r
    tl.atomic_add(out_ptr + tok.to(tl.int64) * K_OUT + p_full, row_part * rw)


def tiny_w4a8mx_moe(
    x, topk_ids, topk_weights, w13_rp, sfb13, w2_rp, sfb2,
    *, n_inter, rot13, inter_buf, out_buf,
):
    M, K = x.shape
    topk = topk_ids.shape[1]
    rt = M * topk
    n2 = 2 * n_inter
    flat_ids = topk_ids.reshape(-1)
    flat_w = topk_weights.reshape(-1)
    inter = inter_buf[:rt]
    inter.zero_()
    out_buf[:M].zero_()

    w13_words = w13_rp.view(torch.int32).reshape(w13_rp.shape[0], -1)
    w2_words = w2_rp.view(torch.int32).reshape(w2_rp.shape[0], -1)
    sfb13_b = sfb13.view(torch.uint8).reshape(sfb13.shape[0], -1)
    sfb2_b = sfb2.view(torch.uint8).reshape(sfb2.shape[0], -1)

    kt13 = K // 128
    KT1 = 1
    _tiny_fc1_kernel[(rt, n2 // 256, kt13 // KT1)](
        x, flat_ids, w13_words, sfb13_b, inter,
        TOPK=topk, K=K, N2=n2,
        W13_STRIDE=w13_words.shape[1], SFB13_STRIDE=sfb13_b.shape[1],
        ROT=rot13, KT_TILES=kt13, KT_PER_PROG=KT1,
        num_warps=8,
    )
    kt2 = n_inter // 128
    KT2 = 1
    _tiny_fc2_kernel[(rt, K // 256, kt2 // KT2)](
        inter, flat_ids, flat_w, w2_words, sfb2_b, out_buf,
        TOPK=topk, N=n_inter, K_OUT=K,
        W2_STRIDE=w2_words.shape[1], SFB2_STRIDE=sfb2_b.shape[1],
        KT_TILES=kt2, KT_PER_PROG=KT2,
        num_warps=8,
    )
    return out_buf[:M].to(torch.bfloat16)
