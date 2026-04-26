# RTX PRO 6000 Blackwell (Workstation Edition) — Power-Limit vs Clock / Performance / Temperature

Power-limit sweep on a single NVIDIA RTX PRO 6000 Blackwell **Workstation Edition** card running `gpu_burn -tc` (tensor-core burn). Power limit set in 50 W intervals from the card's minimum (150 W) to maximum (600 W). At each setting the card was given ~10 s to ramp, then ~30 s of telemetry sampling at 1 Hz; values below are means over the steady-state portion of each window.

Originally requested by Luke (Quant Creators / `lukeRole`) for a clock-vs-power-limit characterisation on Blackwell WS silicon.

---

## System under test

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX PRO 6000 Blackwell **Workstation** Edition |
| GPU index | 3 (selected — clean AER counters; see below) |
| Driver | 595.58.03 |
| CUDA | 13.2 |
| Kernel | Linux 6.18.24 |
| Power limit min / default / max | **150 W / 600 W / 600 W** |
| Memory clock (constant) | 13 365 MHz GDDR7 |
| Burn workload | `gpu_burn -i 3 -tc <duration>` (tensor-core compare) |
| Sampling | `nvidia-smi --query-gpu=...` 1 Hz, average of last 25 of 30 samples |
| Persistence mode | Enabled |
| AER baseline on this GPU | All zeros (no link errors during burn either) |

Note: the **Server Edition** of the same card has a higher minimum power limit (300 W) and is otherwise the same silicon — this sweep was run on a WS card so it covers the full 150 W → 600 W range. A separate Server-edition sweep would only cover 300 W → 600 W.

---

## Results

| Power limit (W) | Actual draw (W) | pstate | GFX / SM clock (MHz) | Mem clock (MHz) | Temp (°C) | Tensor-core throughput (Gflop/s) |
|---:|---:|:-:|---:|---:|---:|---:|
| 150 | 150.0 | P1 |  606 | 13365 | 31.3 |  37 691 |
| 200 | 200.0 | P1 |  657 | 13365 | 37.3 |  59 045 |
| 250 | 250.0 | P1 |  858 | 13365 | 42.0 |  76 315 |
| 300 | 300.0 | P1 | 1031 | 13365 | 46.3 |  91 076 |
| 350 | **342.6** | P1 | 1274 | 13365 | 50.4 | 106 441 |
| 400 | 400.0 | P1 | 1375 | 13365 | 54.1 | 120 203 |
| 450 | **441.7** | P1 | 1577 | 13365 | 57.2 | 132 732 |
| 500 | 499.6 | P1 | 1708 | 13365 | 60.4 | 144 235 |
| 550 | 550.3 | P1 | 1804 | 13365 | 64.3 | 155 836 |
| 600 | 596.3 | P1 | 1948 | 13365 | 67.4 | 165 682 |

The card stays in **pstate P1** the entire time (P0 is reserved for graphics workloads). Memory clock is pinned at 13 365 MHz and never moves regardless of power limit. Two power-limit settings (350 W and 450 W) produced an actual draw lower than the configured limit, suggesting a frequency / voltage step boundary in the boost table where the card cannot use the full extra headroom.

---

## Charts (ASCII, monospace)

### GFX clock vs power limit

```
   GFX MHz
    2000 │                                                          *
    1800 │                                                    *
    1600 │                                              *
    1400 │                                        *
    1200 │                                  *
    1000 │                            *
     800 │                      *
     600 │ *           *
     400 │
     200 │
       0 └─────────────────────────────────────────────────────────────
         150  200  250  300  350  400  450  500  550  600   Power-limit (W)
```

Below ~250 W the card sits at idle-pstate boost-floor (~600 MHz) — additional power headroom does not increase frequency because compute units are already throttled by power. Above ~250 W frequency scales roughly linearly with power.

### Tensor-core throughput vs power limit

```
   Gflop/s
   170k │                                                          *
   150k │                                                    *
   140k │                                              *
   120k │                                        *
   110k │                                  *
   90k  │                            *
   75k  │                      *
   60k  │                *
   40k  │ *           
   20k  │
       └─────────────────────────────────────────────────────────────
         150  200  250  300  350  400  450  500  550  600   Power-limit (W)
```

### Tensor-core efficiency: Gflop/s per Watt

```
   Gflop/s/W
       300 │           *  *  *
       290 │     *           *  *
       280 │ *                       *
       270 │                            *
       260 │                               *  *
       250 │                                     *
       240 │
           └───────────────────────────────────────────────────────
              150  200  250  300  350  400  450  500  550  600   PL (W)
```

Numerically: efficiency in Gflop/s/W (= throughput / actual draw) at each setting:

| PL (W) | Gflop/s | actual W | **efficiency** |
|---:|---:|---:|---:|
| 150 |  37 691 | 150.0 | **251** |
| 200 |  59 045 | 200.0 | **295** |
| 250 |  76 315 | 250.0 | **305** |
| 300 |  91 076 | 300.0 | **304** |
| 350 | 106 441 | 342.6 | **311** |
| 400 | 120 203 | 400.0 | **301** |
| 450 | 132 732 | 441.7 | **300** |
| 500 | 144 235 | 499.6 | **289** |
| 550 | 155 836 | 550.3 | **283** |
| 600 | 165 682 | 596.3 | **278** |

**Sweet spot: ~250–350 W**, where the card delivers ~300–311 Gflop/s/W.
Below 250 W the card cannot fully use the available power (clock-floor sat).
Above 400 W marginal returns drop quickly — going from 400 W to 600 W gains +49 % power but only +38 % throughput.

### Temperature vs power limit (steady state)

```
   °C
    70 │                                                          *
    65 │                                                    *
    60 │                                              *
    55 │                                        *
    50 │                                  *
    45 │                            *
    40 │                      *
    35 │                *
    30 │ *           *
    25 │
       └────────────────────────────────────────────────────────────
         150  200  250  300  350  400  450  500  550  600   Power-limit (W)
```

Temperature scales roughly linearly with actual draw; the card never thermal-throttled in this test (no temp-throttle pstate transitions, never above 67.5 °C). Air cooling, ambient ~22 °C.

---

## Observations / take-aways

1. **WS card honours all 10 power limits** and stays in pstate P1. There is no abrupt cliff or refusal to run at low PL — the card cleanly clocks down to ~600 MHz at 150 W and runs at ~1948 MHz at 600 W.
2. **Memory clock is rock-solid at 13 365 MHz** across the entire sweep. The power limit only gates GFX/SM clock, not memory.
3. **Two limits (350 W and 450 W) were undershot** by the actual draw (343 W and 442 W respectively). The card hit a clock/voltage step before the configured power budget; this looks like a coarse boost table on Blackwell, not a tuning bug.
4. **Best perf/watt is around 300–350 W** at ~310 Gflop/s/W. At default 600 W you only get 278 Gflop/s/W — the last 200 W of headroom is buying clock at decreasing returns. For deployments that care about energy efficiency, capping at ~350 W loses ~36 % of throughput vs 600 W but gains ~12 % efficiency; capping at ~250 W loses ~54 % of throughput for ~10 % efficiency gain.
5. **Thermals never an issue** at any setting on the test rig (4-fan air cooling, ambient ~22 °C). Card peaks at 67 °C at 600 W.

---

## Raw data

### `luke_pl_sweep_results.csv` (steady-state averages)

```csv
power_limit_w,pstate,clk_graphics_mhz,clk_sm_mhz,clk_mem_mhz,power_draw_w,temp_c,gflops
150W,P1,606,606,13365,150.0,31.3,37691
200W,P1,657,657,13365,200.0,37.3,59045
250W,P1,858,858,13365,250.0,42.0,76315
300W,P1,1031,1031,13365,300.0,46.3,91076
350W,P1,1274,1274,13365,342.6,50.4,106441
400W,P1,1375,1375,13365,400.0,54.1,120203
450W,P1,1577,1577,13365,441.7,57.2,132732
500W,P1,1708,1708,13365,499.6,60.4,144235
550W,P1,1804,1804,13365,550.3,64.3,155836
600W,P1,1948,1948,13365,596.3,67.4,165682
```

### Per-second telemetry (300 samples, 30 s × 10 power levels)

`luke_pl_raw_combined.csv` — full Hz-sampled telemetry, 300 rows. First 5:

```csv
power_limit_w,timestamp,gpu_index,pstate,clk_graphics_mhz,clk_sm_mhz,clk_mem_mhz,power_draw_w,temp_c
150,1777238432.400,3,P1,607,607,13365,149.98,30
150,1777238433.666,3,P1,607,607,13365,151.47,29
150,1777238434.887,3,P1,615,615,13365,149.80,30
150,1777238436.108,3,P1,600,600,13365,149.24,30
…
```

Full file kept in the repo under `data/blackwell-ws-pl-sweep.csv`.

---

## Reproduction

Hardware-agnostic; works on any Blackwell-class card. Runtime: ~10 minutes.

```bash
GPU=3            # any single GPU index
GPU_BURN=/path/to/gpu-burn/gpu_burn

nvidia-smi -i $GPU -pm 1
for PL in 150 200 250 300 350 400 450 500 550 600; do
  echo "=== ${PL} W ==="
  nvidia-smi -i $GPU -pl $PL
  sleep 2
  $GPU_BURN -i $GPU -tc 40 > burn_${PL}W.log 2>&1 &
  BURN=$!
  sleep 10                              # ramp
  for i in $(seq 30); do                # 30 s sampling
    nvidia-smi -i $GPU \
      --query-gpu=index,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu \
      --format=csv,noheader | tee -a tel_${PL}W.csv
    sleep 1
  done
  wait $BURN
done
nvidia-smi -i $GPU -pl 600   # restore
```

The full driver script used here is at `/tmp/luke_pl_sweep.sh` on the test rig.

---

## What this does NOT measure

* **Real-application throughput** — `gpu_burn -tc` is a synthetic tensor-core stress, not LLM inference. Real LLM workloads will sit somewhere between this and pure memory-bound regimes; their power-limit response will be different (more flat at low end, since memory clock is invariant).
* **Server-edition silicon** — the Server card has a 300 W floor and may have different boost-table shape. Worth running the same sweep on a Server card for direct comparison.
* **Thermal-limited behaviour** — the rig had plenty of cooling. A passively-cooled or thermally constrained chassis will see throttle at a different power level.
