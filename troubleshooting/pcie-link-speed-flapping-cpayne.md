# PCIe Link Speed Flapping Behind c-payne Switches — Locking Gen5 Without Trapping a Card

RTX PRO 6000 Blackwell cards renegotiate their PCI Express (PCIe) link speed
constantly when idle or lightly loaded. Behind a c-payne Microchip Switchtec
switch this is not merely cosmetic: every renegotiation is a link down/up cycle
on a cabled fabric, and cabled Gen5 fabrics are where marginal signalling shows
up first.

This page documents how to stop the flapping by setting the two PCI Express
capability bits that disable autonomous link changes, and — more importantly —
how to do it **without** locking a card at a speed below its maximum. Getting
that ordering wrong silently costs a working card three quarters of its
bandwidth.

Measured on the ASRockRack GENOAD24QM32-2L2T/BCM platform described in
[`hardware/asrockrack-turin-cpayne-16gpu.md`](../hardware/asrockrack-turin-cpayne-16gpu.md).

Related pages:

- [`hardware/asrockrack-turin-cpayne-16gpu.md`](../hardware/asrockrack-turin-cpayne-16gpu.md) — the same board and switch fabric, peer-to-peer bandwidth characterisation.
- [`hardware/pcie-bandwidth.md`](../hardware/pcie-bandwidth.md) — what a healthy link should deliver.
- [`troubleshooting/gpu-bus-drops-multi-psu-grounding.md`](gpu-bus-drops-multi-psu-grounding.md) — a different failure mode on the same class of frame; worth ruling out first if cards disappear entirely.

## Table of Contents

- [System Under Test](#system-under-test)
- [Symptoms](#symptoms)
- [What The Lock Actually Is](#what-the-lock-actually-is)
- [The Trap: Locking At Enumeration](#the-trap-locking-at-enumeration)
- [The Fix: Verify, Then Lock](#the-fix-verify-then-lock)
- [Manual Procedure](#manual-procedure)
- [Verification](#verification)
- [Pitfalls](#pitfalls)
- [Script](#script)

## System Under Test

| Component | Configuration |
|---|---|
| Motherboard | ASRockRack GENOAD24QM32-2L2T/BCM (single-socket SP5) |
| CPU | AMD EPYC, 8 PCIe root complexes (bus `00`, `20`, `40`, `60`, `80`, `a0`, `c0`, `e0`) |
| GPUs | 17× NVIDIA RTX PRO 6000 Blackwell Workstation, 96 GB |
| PCIe fabric | 8× c-payne Microchip Switchtec Gen5 switch boards, PCI ID `1f18:0101` |
| Uplink | 1× Gen5 ×16 per switch board over MCIO cabling, with retimers |
| Operating system | Ubuntu 24.04, kernel 7.0.0 |

Each switch board carries two GPUs; four of the eight boards carry a third
device (an NVMe drive or a network adapter) on a Gen4/Gen5 ×4 downstream port.

## Symptoms

The flapping itself does not raise an error. It is visible only if you sample
`current_link_speed` faster than the cards change it:

| Observation | Value |
|---|---|
| Link transitions across 35 devices, 2-minute window, cards idle | **1415** |
| Transition pattern | 32 GT/s → 5 GT/s → 2.5 GT/s → 32 GT/s, repeating |
| Devices involved | Every GPU and its parent downstream switch port |
| Transitions in the same window after the lock was applied | **0** |

Sampling `/sys/bus/pci/devices/*/current_link_speed` once per second is enough
to see it. A single `nvidia-smi` or `lspci` snapshot is not — it catches one
phase of the cycle and looks normal, which is exactly how this gets missed.

Why it matters on this fabric:

- Correctable Physical Layer receiver errors (`RxErr`) accumulate on the switch
  downstream ports, because every renegotiation retrains the link.
- The Gen1↔Gen5 retrain is the documented trigger for Surprise Link Down on
  these systems. See
  [Surprise Link Down](common-issues.md#surprise-link-down): the root port can
  suspend *during* a retrain, which surfaces as
  `aer_uncor_status: 0x00000020` and a system lockup.

That entry prescribes the two kernel parameters that stop the root port from
suspending. This page is the complementary half — stopping the retrain from
happening in the first place. Apply both:

```bash
# /etc/default/grub — from the Surprise Link Down entry
GRUB_CMDLINE_LINUX_DEFAULT="pcie_aspm=off pcie_port_pm=off"
```

```bash
# /etc/modprobe.d/nvidia.conf — stop the driver re-enabling runtime D3
options nvidia NVreg_DynamicPowerManagement=0x00
```

One caution on attribution: on the reference system, Surprise Link Down events
were also produced by a genuine power-delivery fault unrelated to link
training. Stopping the flapping is worth doing regardless — a Gen5 link that
renegotiates thousands of times an hour has no business doing so under
production load — but do not assume it is the only cause of a link-down event
you are chasing.

## What The Lock Actually Is

There is no lock feature. It is **two bits in two registers of the PCI Express
capability**, and they must be set on **both ends** of every link.

| Register | `setpci` name | Bit | Meaning |
|---|---|---|---|
| Link Control 2 | `CAP_EXP+30.w` | 5 | Hardware Autonomous Speed Disable — this is the lock |
| Link Control 2 | `CAP_EXP+30.w` | 3:0 | Target Link Speed; `5` = Gen5 |
| Link Control | `CAP_EXP+10.w` | 9 | Hardware Autonomous Width Disable |
| Link Control | `CAP_EXP+10.w` | 5 | Retrain Link — a self-clearing write-only trigger |
| Link Control | `CAP_EXP+10.w` | 1:0 | Active State Power Management (ASPM); keep at `00` |

Reading the two registers on a locked card:

```console
$ setpci -s e3:00.0 CAP_EXP+30.w
0025                       # bit 5 set, target speed 5 (Gen5)
$ setpci -s e3:00.0 CAP_EXP+10.w
0240                       # bit 9 set, ASPM off
```

| `CAP_EXP+30.w` | State |
|---|---|
| `0025` | Locked, target Gen5 |
| `0005` | Unlocked, target Gen5 |

Both ends means the GPU **and** its parent downstream port on the switch.
Setting only one end produces a half-applied lock and confusing behaviour:

```console
$ basename $(dirname $(realpath /sys/bus/pci/devices/0000:e3:00.0))
0000:e2:00.0
```

## The Trap: Locking At Enumeration

The obvious implementation is a udev rule that sets the bits when the device
appears, before the NVIDIA driver binds, while the link is freshly trained:

```udev
# DO NOT USE — see below
ACTION=="add", SUBSYSTEM=="pci", KERNEL=="*:*:*.0", ATTR{vendor}=="0x10de", DRIVER=="", \
    RUN+="/usr/bin/setpci -s %k CAP_EXP+30.w=0x0025:0x003f", \
    RUN+="/usr/bin/setpci -s %k CAP_EXP+10.w=0x0200:0x0203"
```

This does stop the flapping. It also has no way to check what speed the link
actually reached, and **Hardware Autonomous Speed Disable blocks movement in
both directions**. A card that is not at Gen5 at that instant can never climb
back.

> [!WARNING]
> On this system that rule left GPU `e3:00.0` at **Gen2 ×16 for three and a half
> hours of production training** — roughly 8 GB/s instead of 64 GB/s — while
> every other card sat at Gen5. Nothing logged an error. The card reported 100 %
> utilisation the whole time. It was found only by walking every link's
> negotiated speed and comparing it against what the link was capable of.

A udev rule fires once, at a moment you do not control, with no feedback loop.
That is the wrong shape for this problem.

## The Fix: Verify, Then Lock

Move the lock into a small supervisor that runs periodically and holds one
invariant:

> A link may be locked only after it has been verified at its maximum. Nothing
> else is ever locked.

Per pass, for each GPU:

| Link state | Action |
|---|---|
| At its maximum | Lock it, if not already locked. |
| Below maximum | **Unlock** it and allow 20 s to climb back on its own. |
| …came back up | Lock it. |
| …still low, card idle | Force a retrain on the parent port, then lock. |
| …still low, card busy | Leave it **unlocked**, log a warning, retry next pass. |

The last row matters. A card running at Gen2 is worse than a card that flaps, so
a degraded link is never left locked — but a forced retrain under load is worse
still, so the repair waits for the card to go idle.

With a 60-second interval, the worst case after a boot is one link below its
maximum for a minute, instead of indefinitely.

A first pass on a healthy system, with one idle card that had dropped to Gen1:

```console
$ systemctl status pcie-link-supervisor
[pcie-link-sup] 0000:03:00.0 gen5 x16 at maximum -> locked
[pcie-link-sup] 0000:04:00.0 gen5 x16 at maximum -> locked
...
[pcie-link-sup] 0000:c5:00.0 gen5 x4 at maximum -> locked
[pcie-link-sup] 0000:e3:00.0 gen5 x16 at maximum -> locked
[pcie-link-sup] 0000:e4:00.0 BELOW MAXIMUM: gen1 x16 (link supports gen5 x16), locked=False -> unlocking
[pcie-link-sup] 0000:e4:00.0 idle, forcing retrain of port 0000:e2:01.0
[pcie-link-sup] 0000:e4:00.0 after retrain gen5 x16 -> locked
```

## Manual Procedure

> [!WARNING]
> Stop the supervisor first. It re-checks every 60 seconds and will undo manual
> changes: `systemctl stop pcie-link-supervisor`.

`setpci` takes `value:mask`, so only the masked bits are touched.

```bash
GPU=e3:00.0
PORT=$(basename $(dirname $(realpath /sys/bus/pci/devices/0000:$GPU)) | sed 's/0000://')

# 1. Unlock: target speed Gen5 (bits 3:0 = 5), clear Hardware Autonomous Speed Disable (bit 5)
setpci -s $GPU  CAP_EXP+30.w=0x0005:0x003f
setpci -s $PORT CAP_EXP+30.w=0x0005:0x003f
setpci -s $GPU  CAP_EXP+10.w=0x0000:0x0200     # clear Hardware Autonomous Width Disable
setpci -s $PORT CAP_EXP+10.w=0x0000:0x0200

# 2. A loaded card usually returns to Gen5 within seconds on its own.
#    An idle card needs a directed retrain. The Retrain Link bit goes on the
#    PORT, not the GPU, and clears itself.
setpci -s $PORT CAP_EXP+10.w=0x0020:0x0020
sleep 3
cat /sys/bus/pci/devices/0000:$GPU/current_link_speed

# 3. Only once the speed is confirmed, lock.
setpci -s $GPU  CAP_EXP+30.w=0x0020:0x0020
setpci -s $PORT CAP_EXP+30.w=0x0020:0x0020
setpci -s $GPU  CAP_EXP+10.w=0x0200:0x0200
setpci -s $PORT CAP_EXP+10.w=0x0200:0x0200

systemctl start pcie-link-supervisor
```

## Verification

```bash
# every card's negotiated speed and width
nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current \
           --format=csv,noheader

# is the lock bit set?
for g in $(lspci -Dn | grep 10de: | awk '{print $1}' | grep '\.0$'); do
    printf '%s LnkCtl2=%s LnkCtl=%s\n' "${g#0000:}" \
        "$(setpci -s ${g#0000:} CAP_EXP+30.w)" "$(setpci -s ${g#0000:} CAP_EXP+10.w)"
done

# watch for flapping: sample once per second and look for changes
watch -n1 'cat /sys/bus/pci/devices/0000:e3:00.0/current_link_speed'
```

Speed field encoding, for reading `CAP_EXP+30.w` by hand:

| Bits 3:0 | GT/s | Name | ×16 bandwidth |
|---|---|---|---|
| `1` | 2.5 | Gen1 | ~4 GB/s |
| `2` | 5.0 | Gen2 | ~8 GB/s |
| `3` | 8.0 | Gen3 | ~16 GB/s |
| `4` | 16.0 | Gen4 | ~32 GB/s |
| `5` | 32.0 | Gen5 | ~64 GB/s |

## Pitfalls

**A narrow link is not always a fault.** `max_link_width` on an endpoint reports
what the *card* is capable of, not how many lanes are physically wired. On this
board GPU `c5:00.0` reports ×16 but sits on a ×4 downstream port, so **Gen5 ×4
is its maximum and it is healthy**. Always compare the negotiated width against
`min(endpoint maximum, port maximum)`. Comparing against the card's own maximum
makes four devices look permanently degraded and sends any repair loop chasing
them forever.

**An idle Blackwell card at Gen1 is normal.** It is power management, not a
fault. It matters only if something then locks it there.

**Nothing survives a reboot.** PCI configuration space resets on every power
cycle, so the lock must be reapplied by a service or a udev rule each boot. That
is precisely why the naive udev rule is tempting — and why it must not be used.

**Check both ends when debugging.** If a link behaves oddly, read the registers
on the GPU *and* on its parent port. A mismatch between the two is a strong hint
that something applied the lock to only one side.

## Script

[`scripts/pcie-link-supervisor.py`](../scripts/pcie-link-supervisor.py) implements
the verify-then-lock loop described above. It has no dependencies beyond
`python3`, `lspci`, `setpci` and `nvidia-smi`.

```bash
# show what it would do, without changing anything
sudo ./scripts/pcie-link-supervisor.py --interval 0 --dry-run --verbose

# one pass and exit
sudo ./scripts/pcie-link-supervisor.py --interval 0

# run as a service
sudo ./scripts/pcie-link-supervisor.py --interval 60
```

A systemd unit:

```ini
[Unit]
Description=Hold GPU PCIe links at Gen5, never locking below maximum
After=multi-user.target nvidia-persistenced.service

[Service]
Type=simple
ExecStart=/usr/local/bin/pcie-link-supervisor.py --interval 60
Restart=always
RestartSec=15
Nice=10

[Install]
WantedBy=multi-user.target
```
