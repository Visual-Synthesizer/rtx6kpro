"""rp-native tiny-M MoE for w4a8_mx — v4: single fused FC1+SiLU+FC2 kernel.

Program = (routed row, channel block of 32 intermediate channels). Each program:
  1. dots gate+up rows of its 32 channels over the FULL k (acc in registers),
  2. act = silu(gate) * up  (32 registers),
  3. accumulates its channels' FC2 contribution into out via fp32 atomics.
No inter-program dependencies; every weight byte read exactly once per routed row.

Layout exploitation (DS4: K=4096, N=1024, rot13=N):
  w13 logical rows: gate c -> p = (c - rot) mod 2N ... with vLLM w31+rot=N and the
  bench up-first+rot=0 alike, rp tile nt in [0,4) holds the rows whose FC1 role is
  "up" for channels [nt*256,(nt+1)*256) and tile nt+4 holds their "gate" rows
  (verified empirically via the oracle: cos ~ 1).
  Channel-block coords: c_local = n8c*32 + v*8 + r8; the FC1 sub-tile for fixed n8c
  is contiguous per k32 (128 words), and the FC2 column slice for fixed k32 is a
  contiguous 1024-word run. FC1 channel (v, r8) == FC2 channel (cgrp, j).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_val(nib):
    s = (nib >> 3) & 1
    e = (nib >> 1) & 3
    m = nib & 1
    bits = tl.where(e > 0, ((e + 126) << 23) | (m << 22), m * (126 << 23))
    bits = bits | (s << 31)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _tiny_fused_kernel(
    x_ptr, ids_ptr, tw_ptr, w13_ptr, sfb13_ptr, w2_ptr, sfb2_ptr, out_ptr,
    TOPK: tl.constexpr,
    K: tl.constexpr,            # hidden size (fc1 reduction, fc2 output rows)
    N: tl.constexpr,            # intermediate per shard
    W13_STRIDE: tl.constexpr, SFB13_STRIDE: tl.constexpr,
    W2_STRIDE: tl.constexpr, SFB2_STRIDE: tl.constexpr,
    GATE_TILE_OFF: tl.constexpr,  # nt offset of gate rows (N // 256)
    KT13: tl.constexpr,         # K // 128
    KT2: tl.constexpr,          # N // 128
    NT2: tl.constexpr,          # K // 256 (fc2 output row tiles)
):
    # grid: (M*topk, N//256, 8)  -- axis1: channel tile, axis2: n8c group
    pid_rt = tl.program_id(0)
    ct = tl.program_id(1)       # channel tile in [0, N//256)
    g8 = tl.program_id(2)       # n8c group in [0, 8)

    eid = tl.load(ids_ptr + pid_rt).to(tl.int64)
    tok = pid_rt // TOPK
    rw = tl.load(tw_ptr + pid_rt)
    wb13 = eid * W13_STRIDE
    sb13 = eid * SFB13_STRIDE
    wb2 = eid * W2_STRIDE
    sb2 = eid * SFB2_STRIDE

    # ---------------- FC1: 32 up rows (tile ct) + 32 gate rows (tile ct+GATE_TILE_OFF)
    # Load KTC kt-tiles per step; innermost axis is a contiguous 128-word arange
    # (axis-separable affine => vectorized 128-bit loads).
    KTC: tl.constexpr = 4
    kt4 = tl.arange(0, KTC)[:, None, None]
    k32l = tl.arange(0, 4)[None, :, None]
    lane = tl.arange(0, 128)[None, None, :]      # bits: r8(4-6)|cgrp(2-3)|v(0-1)

    # coord views of the loaded (KTC, 4, 128) block, reshaped to (KTC,4,8,4,4)
    k32c = tl.arange(0, 4)[None, :, None, None, None]
    r8c = tl.arange(0, 8)[None, None, :, None, None]
    cgc = tl.arange(0, 4)[None, None, None, :, None]
    vc = tl.arange(0, 4)[None, None, None, None, :]
    kt4c = tl.arange(0, KTC)[:, None, None, None, None]

    acc_u = tl.zeros((8, 4), dtype=tl.float32)   # (r8, v) channel partials
    acc_g = tl.zeros((8, 4), dtype=tl.float32)
    for kto in tl.range(0, KT13 // KTC, num_stages=3):
        col0 = kto * KTC
        base_u = wb13 + (ct * KT13 + col0).to(tl.int64) * 4096 + g8 * 128
        base_g = wb13 + ((ct + GATE_TILE_OFF) * KT13 + col0).to(tl.int64) * 4096 + g8 * 128
        off = kt4 * 4096 + k32l * 1024 + lane
        wu = tl.reshape(tl.load(w13_ptr + base_u + off), (KTC, 4, 8, 4, 4))
        wg = tl.reshape(tl.load(w13_ptr + base_g + off), (KTC, 4, 8, 4, 4))
        # scale: col = (col0+kt)*4 + k32; row bits (n8c=g8, v, r8)
        sfo_u = k32c + r8c * 4 + (g8 * 4 + vc) * 32 + (ct * KT13 + col0 + kt4c) * 1024
        sfo_g = k32c + r8c * 4 + (g8 * 4 + vc) * 32 + (((ct + GATE_TILE_OFF) * KT13 + col0 + kt4c)) * 1024
        su = (tl.load(sfb13_ptr + sb13 + sfo_u).to(tl.int32) << 23).to(tl.float32, bitcast=True)
        sg = (tl.load(sfb13_ptr + sb13 + sfo_g).to(tl.int32) << 23).to(tl.float32, bitcast=True)
        pu = tl.zeros((KTC, 4, 8, 4, 4), dtype=tl.float32)
        pg = tl.zeros((KTC, 4, 8, 4, 4), dtype=tl.float32)
        for j in tl.static_range(8):
            a = tl.load(
                x_ptr + tok * K + (col0 + kt4c) * 128 + (k32c * 4 + cgc) * 8 + j
            ).to(tl.float32)
            pu += _fp4_val((wu >> (4 * j)) & 0xF) * a
            pg += _fp4_val((wg >> (4 * j)) & 0xF) * a
        acc_u += tl.sum(tl.sum(tl.sum(pu * su, axis=3), axis=1), axis=0)
        acc_g += tl.sum(tl.sum(tl.sum(pg * sg, axis=3), axis=1), axis=0)

    act = (acc_g / (1.0 + tl.exp(-acc_g))) * acc_u * rw   # (r8, v); router weight folded

    # ---------------- FC2: columns = my 32 channels; c0 = ct*256 + g8*32
    # word w = c0/8 + cgrp2, i.e. kt2c = c0>>7, k32c = (c0>>5)&3; channel (cgrp2, j).
    c0 = ct * 256 + g8 * 32
    kt2c = c0 // 128
    k32c_s = (c0 // 32) % 4
    n8c2 = tl.arange(0, 8)[:, None, None, None]
    r82 = tl.arange(0, 8)[None, :, None, None]
    cg2 = tl.arange(0, 4)[None, None, :, None]
    v2 = tl.arange(0, 4)[None, None, None, :]

    j_sel = tl.arange(0, 8)[:, None]
    for nt2 in tl.range(0, NT2, num_stages=3):
        base2 = wb2 + (nt2 * KT2 + kt2c).to(tl.int64) * 4096 + k32c_s * 1024
        w2w = tl.reshape(tl.load(w2_ptr + base2 + tl.arange(0, 1024)), (8, 8, 4, 4))
        sfo2 = k32c_s + r82 * 4 + (n8c2 * 4 + v2) * 32 + (nt2 * KT2 + kt2c) * 1024
        s2 = (tl.load(sfb2_ptr + sb2 + sfo2).to(tl.int32) << 23).to(tl.float32, bitcast=True)
        po = tl.zeros((8, 8, 4, 4), dtype=tl.float32)
        for j in tl.static_range(8):
            # act channel for (cgrp2=cg2, j): act[(r8==j, v==cg2)]
            a_j = tl.sum(tl.where(j_sel == j, act, 0.0), axis=0)      # (4,) over v
            a_b = a_j[None, None, :, None]                            # broadcast to (1,1,4,1)
            po += _fp4_val((w2w >> (4 * j)) & 0xF) * a_b
        acc_o = tl.sum(po * s2, axis=2)                               # (8, 8, 4) rows
        n8c_o = tl.arange(0, 8)[:, None, None]
        r8_o = tl.arange(0, 8)[None, :, None]
        v_o = tl.arange(0, 4)[None, None, :]
        p2 = nt2 * 256 + n8c_o * 32 + v_o * 8 + r8_o
        tl.atomic_add(out_ptr + tok.to(tl.int64) * K + p2, acc_o)


def tiny_w4a8mx_moe_v4(
    x, topk_ids, topk_weights, w13_rp, sfb13, w2_rp, sfb2,
    *, n_inter, out_buf,
):
    M, K = x.shape
    topk = topk_ids.shape[1]
    rt = M * topk
    flat_ids = topk_ids.reshape(-1)
    flat_w = topk_weights.reshape(-1)
    out_buf[:M].zero_()

    w13_words = w13_rp.view(torch.int32).reshape(w13_rp.shape[0], -1)
    w2_words = w2_rp.view(torch.int32).reshape(w2_rp.shape[0], -1)
    sfb13_b = sfb13.view(torch.uint8).reshape(sfb13.shape[0], -1)
    sfb2_b = sfb2.view(torch.uint8).reshape(sfb2.shape[0], -1)

    _tiny_fused_kernel[(rt, n_inter // 256, 8)](
        x, flat_ids, flat_w, w13_words, sfb13_b, w2_words, sfb2_b, out_buf,
        TOPK=topk, K=K, N=n_inter,
        W13_STRIDE=w13_words.shape[1], SFB13_STRIDE=sfb13_b.shape[1],
        W2_STRIDE=w2_words.shape[1], SFB2_STRIDE=sfb2_b.shape[1],
        GATE_TILE_OFF=n_inter // 256,
        KT13=K // 128, KT2=n_inter // 128, NT2=K // 256,
        num_warps=8,
    )
    return out_buf[:M].to(torch.bfloat16)
