# Intermittent GPU Bus Drops on Multi-PSU Open-Air Frames

This page records a community field report in which GPUs intermittently
disappeared from a 16-GPU PCIe-switch system. Restoring a compliant protective
earth (PE) connection to the PSU chassis and shared metal frame stopped the
failures for more than 48 hours.

This is a useful diagnostic case, not proof that every `GPU has fallen off the
bus` event is a grounding fault. NVIDIA documents Xid messages as starting
points for investigation, because the same Xid can have hardware, software, or
application causes.

> [!WARNING]
> Mains wiring and protective earth are life-safety systems. Do not insert
> improvised conductors into an outlet, alter a plug, work on energized wiring,
> or create an ad hoc chassis bond. Disconnect the equipment and have a
> qualified electrician correct and verify the installation according to local
> electrical code.

## Reported System

| Component | Configuration |
|---|---|
| Motherboard | ASUS Pro WS WRX90E-SAGE SE |
| CPU | AMD Ryzen Threadripper PRO 9955WX |
| GPUs | 16x NVIDIA RTX PRO 6000 Blackwell Workstation, 96 GB |
| PCIe fabric | 3x c-payne Microchip Switchtec PM50100 Gen5, arranged as 8+4+4 GPUs |
| PSUs | 4x ASUS Pro WS 3000W, mounted to the same metal frame as the GPUs |
| Mains connection | CEE 7/7 plugs; the used outlets did not provide a working PE contact |
| Driver / CUDA | 595.71.05 / CUDA 13.2 |
| Example workloads | GLM-5.2 NVFP4 and Kimi K2.7 Code inference |

Source: [Discord field report, 2026-07-27](https://discord.com/channels/1466898002793857221/1474436297878933699/1531395551088742461).

## Symptoms

- One or more GPUs randomly disappeared from `nvidia-smi`.
- Failures occurred most often under inference load, but occasionally at idle.
- All three PCIe-switch groups were affected rather than one card or one
  switch.
- The observed rate was approximately one to seven failures per day.
- A warm reboot usually restored the GPUs; some events required a complete
  power cycle.
- Kernel logs contained NVIDIA Xid errors, while PCIe AER counters remained
  clear.
- GPU temperatures remained below 80 C.

An Xid code should be recorded exactly. In particular, NVIDIA defines Xid 79 as
`GPU has fallen off the bus`, but the code alone does not identify the root
cause.

## Evidence From The Investigation

The following substitutions did not stop the failures:

| Change | Result |
|---|---|
| Replaced multiple MCIO cables | No change |
| Replaced host-side adapters | No change |
| Replaced device-side adapters | No change |
| Moved GPU cards between positions | Failure did not follow one card |
| Changed the switch layout from 8+8 to 8+4+4 | No change |

Moving the most frequently affected GPU away from the shared metal frame also
changed which card failed. That observation pointed toward the frame's
electrical environment, but it was not a controlled isolation test. Do not
operate a powered GPU on an anti-static bag: its outer surface may be
conductive.

The decisive intervention was restoration of the missing PE path for all four
PSUs. No driver, model, container, kernel parameter, or PCIe component was
changed at the same time. The system then completed more than 48 hours of the
same workload without a GPU drop; before the repair, a failure-free 48-hour
period had not been observed.

This before/after result makes the missing PE connection the strongest cause in
this case. It remains a single-system field report rather than a controlled
electrical measurement campaign.

## Why A Missing PE Connection Matters

Class-I PSUs normally bond their metal chassis to PE through the power cord.
When that path is absent, EMI-filter leakage can leave a chassis at a floating
AC common-mode potential. In an open-frame system, PSU enclosures, GPU
brackets, switch boards, and the frame can also be connected through mounting
hardware and signal grounds. Multiple floating PSUs can therefore create
unwanted common-mode currents or reference differences across the PCIe fabric.

A permanent and continuous equipment-grounding path is primarily a shock and
fault-current safety requirement. Signal integrity or device stability is a
secondary reason to inspect it, never a substitute for electrical-safety
compliance.

Zero AER events do not clear the power and grounding system. AER reports PCIe
protocol/link errors; it does not prove that chassis potentials, PSU leakage,
or every power rail remained within specification.

## Safe Diagnostic Procedure

1. Record the exact failing GPU, PCI BDF, Xid code, workload, temperature, and
   whether the event also occurs at idle.
2. Preserve kernel evidence before rebooting:

   ```bash
   nvidia-smi -L
   nvidia-smi --query-gpu=index,pci.bus_id,uuid,temperature.gpu,power.draw \
     --format=csv
   journalctl -k -b | grep -Ei 'NVRM|Xid|fallen off|AER|PCIe'
   lspci -tv
   ```

3. Inspect normal hardware causes: PCIe link training, MCIO seating, retimers,
   switch power, GPU auxiliary power, PSU loading, thermals, and mechanical
   strain. Change one variable at a time.
4. Fully de-energize and unplug every PSU before inspecting the frame, plugs,
   receptacles, protective conductors, or chassis bonds.
5. Have a qualified electrician verify that each receptacle supplies a valid PE
   path and that exposed conductive parts have the required permanent,
   low-impedance bonding. Outlet polarity alone is not enough.
6. Correct the supply wiring or receptacle. Do not use a loose pin, adapter that
   defeats PE, signal cable, PCIe bracket, or DC-negative lead as the protective
   conductor.
7. Repeat the same sustained workload for at least as long as the previous
   failure interval. Continue logging Xids and AER events so the before/after
   comparison is meaningful.

Do not bond a PSU's DC negative output to PE unless the PSU or system vendor
explicitly requires it. Protective-earth bonding and DC-output grounding are
different design decisions.

## Interpretation Guide

| Observation | What it suggests |
|---|---|
| Failure follows one GPU | Card, connector, or card-specific power path |
| Failure follows one cable, adapter, or switch port | PCIe physical path |
| Multiple switch groups fail, including at idle | Shared power, frame, firmware, or host issue |
| Xid 79 with a missing device | GPU left the PCIe bus; continue hardware triage |
| No AER counters | No recorded PCIe AER event; other causes remain possible |
| Repairing PE is the only change and failures stop | Strong evidence for a grounding/power-domain cause |

## References

- [NVIDIA Xid error documentation](https://docs.nvidia.com/deploy/xid-errors/introduction.html)
- [NVIDIA GPU Debug Guidelines](https://docs.nvidia.com/deploy/gpu-debug-guidelines/index.html)
- [OSHA grounding overview](https://www.osha.gov/etools/construction/electrical-incidents/grounding)
- [Original Discord field report](https://discord.com/channels/1466898002793857221/1474436297878933699/1531395551088742461)

