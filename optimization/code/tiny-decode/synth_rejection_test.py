"""Synthetic kernel-level test for rejection_sample: standard vs block verification.

No model, no server: drive the Triton kernels directly with synthetic
target/draft distributions of controlled divergence. Reports:
  1) correctness: at temp=0 block == standard exactly; output-token
     distribution matches target softmax (distribution preservation)
  2) accepted-length gain of block vs standard across (temp, draft quality)
"""
import sys
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

torch.manual_seed(0)
DEV = "cuda"
K = 5          # num_speculative_steps
V = 4096       # vocab (small for speed; kernels are vocab-blocked anyway)
MAXR = 512


def make_batch(R, temp, eps, seed0):
    """R requests, each k=5 drafts + 1 bonus row."""
    n = R * (K + 1)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed0)
    base = torch.randn(R, K + 1, V, generator=g, device=DEV) * 2.0
    target_logits = base.reshape(n, V).contiguous()
    # draft = target 'view' with noise: higher eps = worse draft
    draft_logits_full = base[:, :K] + eps * torch.randn(R, K, V, generator=g, device=DEV)
    draft_logits = torch.zeros(MAXR, K, V, device=DEV)
    draft_logits[:R] = draft_logits_full
    # draft tokens sampled from q at the request temperature (or argmax)
    if temp > 0:
        q = torch.softmax(draft_logits_full / temp, dim=-1)
        d = torch.multinomial(q.reshape(-1, V), 1, generator=g).reshape(R, K)
    else:
        d = draft_logits_full.argmax(-1)
    draft_sampled = torch.zeros(R, K + 1, dtype=torch.int64, device=DEV)
    draft_sampled[:, 1:] = d  # row 0 = anchor (unused by kernels)
    draft_sampled = draft_sampled.reshape(n).to(torch.int32)

    cu = torch.arange(0, R + 1, device=DEV, dtype=torch.int32) * (K + 1)
    pos = (torch.arange(n, device=DEV, dtype=torch.int64) % (K + 1)) + 100
    idx_mapping = torch.arange(R, device=DEV, dtype=torch.int32)
    exp_idx = idx_mapping.repeat_interleave(K + 1)
    exp_pos = (torch.arange(n, device=DEV) % (K + 1)).to(torch.int32)
    temperature = torch.zeros(MAXR, device=DEV) + temp
    seeds = torch.arange(MAXR, device=DEV, dtype=torch.int64) + seed0 * 7919
    return (target_logits, draft_logits, draft_sampled, cu, pos, idx_mapping,
            exp_idx, exp_pos, temperature, seeds)


def run(R, temp, eps, seed0, block):
    args = make_batch(R, temp, eps, seed0)
    (tl_, dl, ds, cu, pos, im, eim, elp, tt, sd) = args
    # temperature-process target logits like the sampler does (divide by temp)
    proc = tl_ / max(temp, 1e-6) if temp > 0 else tl_
    sampled, num_sampled = rejection_sample(
        proc, dl, ds, cu, pos, im, eim, elp, tt, sd, K,
        None, use_fp64=False, use_block_verification=block,
    )
    return sampled, num_sampled


def accepted_stats(num_sampled):
    # num_sampled = accepted + 1 (bonus/resample)
    acc = (num_sampled.float() - 1).clamp(min=0)
    return acc.mean().item()


print("== 1) temp=0: block must equal standard exactly")
for eps in (0.5, 2.0):
    s1, n1 = run(256, 0.0, eps, 1, False)
    s2, n2 = run(256, 0.0, eps, 1, True)
    # compare only the valid prefix of each row (tail slots are dead memory)
    mask = torch.arange(s1.shape[1], device=s1.device)[None, :] < n1[:, None]
    same = torch.equal(n1, n2) and torch.equal(s1[mask], s2[mask])
    print(f"   eps={eps}: identical={same}, mean_accepted={accepted_stats(n1):.3f}")
    assert same

print("== 2) accepted length: standard vs block across (temp, eps)")
for temp in (0.6, 0.8, 1.0):
    for eps in (0.3, 0.7, 1.2):
        a_std = a_blk = 0.0
        TRIALS = 8
        for t in range(TRIALS):
            _, ns = run(256, temp, eps, 10 + t, False)
            _, nb = run(256, temp, eps, 10 + t, True)
            a_std += accepted_stats(ns) / TRIALS
            a_blk += accepted_stats(nb) / TRIALS
        gain = (a_blk / a_std - 1) * 100
        print(f"   temp={temp} eps={eps}: std={a_std:.3f} block={a_blk:.3f} ({gain:+.1f}%)")

print("== 3) distribution preservation (V=64, first position, chi-square-ish)")
# small-vocab variant inline
def small_check(block, temp=0.8, eps=0.8, trials=400):
    torch.manual_seed(5)
    Vs = 64
    counts = torch.zeros(Vs, device=DEV)
    ref_p = None
    for t in range(trials):
        g = torch.Generator(device=DEV); g.manual_seed(1000 + t)
        base = torch.randn(1, K + 1, Vs, generator=g, device=DEV) * 2.0
        # FIXED target for position 0 across trials: reuse trial-0 logits row 0
        if t == 0:
            fixed_row = base[0, 0].clone()
        base[0, 0] = fixed_row
        tl_ = base.reshape(K + 1, Vs).contiguous()
        dlf = base[:, :K] + eps * torch.randn(1, K, Vs, generator=g, device=DEV)
        dl = torch.zeros(MAXR, K, Vs, device=DEV); dl[:1] = dlf
        q = torch.softmax(dlf / temp, -1)
        d = torch.multinomial(q.reshape(-1, Vs), 1, generator=g).reshape(1, K)
        ds = torch.zeros(1, K + 1, dtype=torch.int64, device=DEV)
        ds[:, 1:] = d
        ds = ds.reshape(-1).to(torch.int32)
        cu = torch.tensor([0, K + 1], device=DEV, dtype=torch.int32)
        pos = torch.arange(K + 1, device=DEV, dtype=torch.int64) + 100 + t * 31
        im = torch.zeros(1, device=DEV, dtype=torch.int32)
        eim = torch.zeros(K + 1, device=DEV, dtype=torch.int32)
        elp = torch.arange(K + 1, device=DEV, dtype=torch.int32)
        tt = torch.zeros(MAXR, device=DEV) + temp
        sd = torch.zeros(MAXR, device=DEV, dtype=torch.int64) + 977 * t
        proc = tl_ / temp
        sampled, ns = rejection_sample(proc, dl, ds, cu, pos, im, eim, elp,
                                       tt, sd, K, None, use_fp64=False,
                                       use_block_verification=block)
        counts[sampled[0, 0].long()] += 1
        if ref_p is None:
            ref_p = torch.softmax(fixed_row / temp, -1)
    emp = counts / counts.sum()
    tvd = 0.5 * (emp - ref_p).abs().sum().item()
    return tvd

for block in (False, True):
    tvd = small_check(block)
    print(f"   {'block' if block else 'std  '}: TVD(empirical, target) = {tvd:.4f} (expect ~< 0.15 at 400 trials)")
print("ALL SYNTHETIC CHECKS DONE")
