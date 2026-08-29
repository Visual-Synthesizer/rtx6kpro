#!/usr/bin/env python3
"""Hold GPU PCIe links at their maximum speed, never locking one below it.

Why this exists
---------------
RTX PRO 6000 Blackwell cards renegotiate their PCI Express link speed constantly
when idle. Behind a c-payne Microchip Switchtec switch that means thousands of
link down/up cycles per hour on a cabled Gen5 fabric: 1415 transitions across 35
devices in a two-minute window were measured on the reference system, cycling
32 -> 5 -> 2.5 -> 32 GT/s.

Setting "Hardware Autonomous Speed Disable" stops that completely. The obvious
place to set it is a udev rule at device enumeration -- and that is a trap. The
bit blocks movement in *both* directions, and a udev rule cannot check what
speed the link actually reached. On the reference system such a rule left one
GPU at Gen2 x16 for three and a half hours of production training, roughly
8 GB/s instead of 64 GB/s, with nothing logging an error.

The invariant this script holds
-------------------------------
    A link may be locked only after it has been verified at its maximum.
    Nothing else is ever locked.

Per pass, for every GPU:
  1) at maximum            -> lock it (if not already locked)
  2) below maximum         -> UNLOCK and let it climb back
     a) returns within RECOVER_WAIT       -> lock
     b) still low and the card is idle    -> force a retrain, then lock
     c) still low and the card is busy    -> leave UNLOCKED and warn
        (running at Gen2 is worse than flapping, but a forced retrain under
         load is worse still, so the repair waits for the card to go idle)

Running periodically means a trapped link is repaired within one interval
instead of indefinitely.

See troubleshooting/pcie-link-speed-flapping-cpayne.md for the full write-up.
"""

import argparse
import os
import re
import subprocess
import sys
import time

LOG = "[pcie-link-sup]"
SYS = "/sys/bus/pci/devices"

NVIDIA_VENDOR = "10de:"

# PCI Express capability register offsets, as setpci names.
LNKCTL = "CAP_EXP+10.w"      # bit 9 = Hardware Autonomous Width Disable
LNKCTL2 = "CAP_EXP+30.w"     # bit 5 = Hardware Autonomous Speed Disable, bits 3:0 = target
SPEED_DIS = 0x0020
WIDTH_DIS = 0x0200
RETRAIN = 0x0020             # bit 5 of LNKCTL, self-clearing
TARGET_MASK = 0x003F
TARGET_GEN5 = 0x0005

RECOVER_WAIT = 20.0          # seconds to wait for an unlocked link to climb back
BUSY_UTIL_PCT = 20           # above this the card counts as in use
GEN_OF = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def read(slot, name):
    try:
        with open(os.path.join(SYS, slot, name)) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def speed(txt):
    m = re.match(r"([\d.]+)", txt or "")
    return float(m.group(1)) if m else 0.0


def gen(value):
    return GEN_OF.get(value, "?")


def gpus():
    """[(gpu_slot, parent_downstream_port_slot), ...] for every NVIDIA function 0."""
    out = sh(["lspci", "-Dn"]) or ""
    res = []
    for line in out.splitlines():
        if NVIDIA_VENDOR not in line:
            continue
        slot = line.split()[0]
        if not slot.endswith(".0"):
            continue
        real = os.path.realpath(os.path.join(SYS, slot))
        res.append((slot, os.path.basename(os.path.dirname(real))))
    return res


def link(slot):
    return (speed(read(slot, "current_link_speed")), int(read(slot, "current_link_width") or 0),
            speed(read(slot, "max_link_speed")), int(read(slot, "max_link_width") or 0))


def effective_max(slot, parent):
    """The lower capability of the two ends of the link.

    max_link_width on an endpoint reports what the *card* can do, not how many
    lanes are physically wired. Without taking the minimum, every x16 card in an
    x4 slot looks permanently degraded and the repair loop chases it forever.
    """
    _, _, own_s, own_w = link(slot)
    _, _, par_s, par_w = link(parent)
    eff_s = min(own_s, par_s) if own_s and par_s else (own_s or par_s)
    eff_w = min(own_w, par_w) if own_w and par_w else (own_w or par_w)
    return eff_s, eff_w


def busy(slot):
    bus = slot.split(":", 1)[1].upper()
    out = sh(["nvidia-smi", "--query-gpu=pci.bus_id,utilization.gpu",
              "--format=csv,noheader,nounits"]) or ""
    for line in out.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 2 and f[0].upper().endswith(bus):
            try:
                return int(f[1]) > BUSY_UTIL_PCT
            except ValueError:
                return False
    return False


def setpci(slot, reg, value, mask):
    sh(["setpci", "-s", slot, f"{reg}={value:04x}:{mask:04x}"])


def get_reg(slot, reg):
    v = sh(["setpci", "-s", slot, reg])
    try:
        return int(v, 16)
    except (TypeError, ValueError):
        return None


def is_locked(slot):
    v = get_reg(slot, LNKCTL2)
    return bool(v is not None and v & SPEED_DIS)


def lock(slot, parent, dry):
    if dry:
        return "dry-run"
    for d in (slot, parent):
        setpci(d, LNKCTL2, SPEED_DIS, SPEED_DIS)
        setpci(d, LNKCTL, WIDTH_DIS, WIDTH_DIS)
    return "locked"


def unlock(slot, parent, dry):
    if dry:
        return "dry-run"
    for d in (slot, parent):
        setpci(d, LNKCTL2, TARGET_GEN5, TARGET_MASK)
        setpci(d, LNKCTL, 0x0000, WIDTH_DIS)
    return "unlocked"


def retrain(parent, dry):
    if not dry:
        setpci(parent, LNKCTL, RETRAIN, RETRAIN)


def pass_once(dry, verbose):
    acted = 0
    for slot, parent in gpus():
        cur_s, cur_w = link(slot)[:2]
        max_s, max_w = effective_max(slot, parent)
        if not cur_s or not max_s:
            continue
        at_max = cur_s >= max_s - 0.01 and (not max_w or cur_w >= max_w)
        locked = is_locked(slot)

        if at_max:
            if not locked:
                acted += 1
                print(f"{LOG} {slot} gen{gen(cur_s)} x{cur_w} at maximum "
                      f"-> {lock(slot, parent, dry)}", flush=True)
            elif verbose:
                print(f"{LOG} {slot} gen{gen(cur_s)} x{cur_w} OK, locked", flush=True)
            continue

        print(f"{LOG} {slot} BELOW MAXIMUM: gen{gen(cur_s)} x{cur_w} "
              f"(link supports gen{gen(max_s)} x{max_w}), locked={locked} -> unlocking",
              flush=True)
        unlock(slot, parent, dry)
        acted += 1
        if dry:
            print(f"{LOG} {slot} dry-run: would wait for recovery, then lock", flush=True)
            continue

        deadline = time.time() + RECOVER_WAIT
        while time.time() < deadline:
            time.sleep(2)
            cur_s, cur_w = link(slot)[:2]
            if cur_s >= max_s - 0.01 and (not max_w or cur_w >= max_w):
                print(f"{LOG} {slot} recovered on its own to gen{gen(cur_s)} x{cur_w}"
                      f" -> {lock(slot, parent, dry)}", flush=True)
                break
        else:
            if busy(slot):
                print(f"{LOG} {slot} still below maximum and the card is BUSY - leaving "
                      f"UNLOCKED. No forced retrain under load; will retry next pass.",
                      flush=True)
            else:
                print(f"{LOG} {slot} idle, forcing retrain of port {parent}", flush=True)
                retrain(parent, dry)
                time.sleep(3)
                cur_s, cur_w = link(slot)[:2]
                if cur_s >= max_s - 0.01:
                    print(f"{LOG} {slot} after retrain gen{gen(cur_s)} x{cur_w}"
                          f" -> {lock(slot, parent, dry)}", flush=True)
                else:
                    print(f"{LOG} {slot} retrain did not help (gen{gen(cur_s)}), "
                          f"leaving UNLOCKED", flush=True)
    return acted


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=60.0,
                    help="seconds between passes; 0 runs a single pass and exits")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, touch nothing")
    ap.add_argument("--verbose", action="store_true",
                    help="also report links that are already correct")
    args = ap.parse_args()

    if os.geteuid() != 0 and not args.dry_run:
        sys.exit(f"{LOG} needs root for setpci")

    print(f"{LOG} start interval={args.interval}s dry_run={args.dry_run} "
          f"invariant: only ever lock a link verified at its maximum", flush=True)
    while True:
        try:
            pass_once(args.dry_run, args.verbose)
        except Exception as exc:                            # noqa: BLE001
            print(f"{LOG} error during pass: {exc}", flush=True)
        if args.interval <= 0:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
