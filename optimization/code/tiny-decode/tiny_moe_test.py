import sys, statistics, torch
sys.path.insert(0, "/bench")
sys.path.insert(0, "/work")
from benchmark_ds4_moe import make_synthetic_mxfp4_moe, _prepare_b12x_experts
from b12x.integration.tp_moe import (
    allocate_tp_moe_workspace_pool, b12x_moe_fp4,
    build_tp_moe_fp4_binding, clear_tp_moe_caches,
)
import benchmark_ds4_moe as B
from tiny_moe import tiny_w4a8mx_moe
from tiny_moe_v4 import tiny_w4a8mx_moe_v4
USE = 'v3'

torch.manual_seed(7)
dev = torch.device("cuda")
E, K, N = 256, 4096, 1024
src = make_synthetic_mxfp4_moe(E, K, N, seed=7, device=dev)
# logical copies for the oracle BEFORE prep mutates anything
w13_log = src["w13_fp4"].clone()
w13_mx = src["w13_mx"].clone()
w2_log = src["w2_fp4"].clone()
w2_mx = src["w2_mx"].clone()

def decode_fp4(u8, grid, rows, k):  # -> f32 [E, rows, k]
    lo = (u8 & 0xF).to(torch.int32)
    hi = ((u8 >> 4) & 0xF).to(torch.int32)
    nib = torch.stack([lo, hi], dim=-1).reshape(u8.shape[0], rows, k)
    mag_lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=u8.device)
    mag = mag_lut[(nib & 7).long()]
    val = torch.where((nib >> 3) == 1, -mag, mag).float()
    scale = torch.exp2(grid.view(u8.shape[0], rows, k // 32).float() - 127.0)
    return val * scale.repeat_interleave(32, dim=2)

W13 = decode_fp4(w13_log, w13_mx, 2 * N, K)   # [E, 2n, k]
W2 = decode_fp4(w2_log, w2_mx, K, N)          # [E, k, n]

clear_tp_moe_caches()
experts = _prepare_b12x_experts("w4a8_mx", src)
w13_rp, sfb13 = experts.w1_fp4, experts.w1_blockscale
w2_rp, sfb2 = experts.w2_fp4, experts.w2_blockscale
print("rp tensors:", tuple(w13_rp.shape), w13_rp.dtype, tuple(sfb13.shape), sfb13.dtype)

inter_buf = torch.zeros(64, 2 * N, dtype=torch.float32, device=dev)
out_buf = torch.zeros(8, K, dtype=torch.float32, device=dev)

for M in (1, 4, 8):
    gen = torch.Generator(device=dev); gen.manual_seed(1000 + M)
    x = (torch.randn(M, K, generator=gen, device=dev) * 2.0).to(torch.bfloat16)
    logits = torch.randn(M, E, generator=gen, device=dev)
    tl_, ti = torch.topk(logits, B.DS4_TOPK, dim=-1)
    tw = torch.softmax(tl_, dim=-1).float(); ti = ti.to(torch.int32)

    # fp32 oracle
    xf = x.float()
    oracle = torch.zeros(M, K, device=dev)
    for m in range(M):
        for j in range(B.DS4_TOPK):
            e = int(ti[m, j]); w = float(tw[m, j])
            h = W13[e] @ xf[m]                      # [2n]
            up, gate = h[:N], h[N:]                 # synthetic weights are [up; gate]
            act = torch.nn.functional.silu(gate) * up
            oracle[m] += w * (W2[e] @ act)

    # dynamic reference
    out_dyn = torch.empty(M, K, dtype=torch.bfloat16, device=dev)
    ws = allocate_tp_moe_workspace_pool()
    binding = build_tp_moe_fp4_binding(scratch=ws, a=x, experts=experts,
        topk_weights=tw, topk_ids=ti, output=out_dyn, input_scales_static=True, quant_mode="w4a8_mx")
    b12x_moe_fp4(binding=binding); torch.cuda.synchronize()

    out_tiny = (tiny_w4a8mx_moe(x, ti, tw, w13_rp, sfb13, w2_rp, sfb2,
                                n_inter=N, rot13=N, inter_buf=inter_buf, out_buf=out_buf)
                if USE == 'v3' else
                tiny_w4a8mx_moe_v4(x, ti, tw, w13_rp, sfb13, w2_rp, sfb2,
                                   n_inter=N, out_buf=out_buf))
    torch.cuda.synchronize()

    def cos(a, b):
        a, b = a.flatten().float(), b.flatten().float()
        return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)).item()
    print(f"M={M}: cos(tiny,oracle)={cos(out_tiny, oracle):.6f}  cos(dyn,oracle)={cos(out_dyn, oracle):.6f}  cos(tiny,dyn)={cos(out_tiny, out_dyn):.6f}")

# ---- performance: graph replay ----
print("\n== graph-replay us/layer (vs dynamic 34.8/88/210, a16 22.5/75.6/190.4)")
for M in (1, 4, 8):
    gen = torch.Generator(device=dev); gen.manual_seed(1000 + M)
    x = (torch.randn(M, K, generator=gen, device=dev) * 2.0).to(torch.bfloat16)
    logits = torch.randn(M, E, generator=gen, device=dev)
    tl_, ti = torch.topk(logits, B.DS4_TOPK, dim=-1)
    tw = torch.softmax(tl_, dim=-1).float(); ti = ti.to(torch.int32)
    for _ in range(3):
        (tiny_w4a8mx_moe(x, ti, tw, w13_rp, sfb13, w2_rp, sfb2,
                         n_inter=N, rot13=N, inter_buf=inter_buf, out_buf=out_buf)
         if USE == 'v3' else
         tiny_w4a8mx_moe_v4(x, ti, tw, w13_rp, sfb13, w2_rp, sfb2, n_inter=N, out_buf=out_buf))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        (tiny_w4a8mx_moe(x, ti, tw, w13_rp, sfb13, w2_rp, sfb2,
                         n_inter=N, rot13=N, inter_buf=inter_buf, out_buf=out_buf)
         if USE == 'v3' else
         tiny_w4a8mx_moe_v4(x, ti, tw, w13_rp, sfb13, w2_rp, sfb2, n_inter=N, out_buf=out_buf))
    g.replay(); torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    times = []
    for _ in range(5):
        s.record()
        for _ in range(50):
            g.replay()
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 50 * 1000)
    print(f"M={M}: {statistics.median(times):7.1f} us/layer")
