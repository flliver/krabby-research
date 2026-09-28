"""Critic-reset surgery for the phased-flat B1 arm.

Loads an RSL-RL checkpoint, re-initializes every `critic.*` parameter with fresh
nn.Linear-default init (Kaiming-uniform weights, fan-in-uniform bias), clears the
optimizer state entirely (fresh Adam moments for actor and critic), and writes a new
checkpoint. Same procedure as the 2026-08-17 falsification test on the v2 base
(parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/lit-review-plasticity-cliff.md §5); here applied to the plastic 3k base at the
Phase-B boundary where the reward stack changes (reversal -0.3 turned on).
"""

import argparse
import math

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    msd = ckpt["model_state_dict"]

    critic_keys = [k for k in msd if k.startswith("critic.")]
    if not critic_keys:
        raise SystemExit(f"no critic.* keys in {args.src}: {sorted(msd)[:10]}")

    before = torch.norm(torch.cat([msd[k].flatten() for k in critic_keys])).item()
    for k in critic_keys:
        t = msd[k]
        if t.dim() >= 2:  # weight
            torch.nn.init.kaiming_uniform_(t, a=math.sqrt(5))
        else:  # bias: uniform bound from the matching weight's fan-in
            wkey = k.rsplit(".", 1)[0] + ".weight"
            fan_in = msd[wkey].shape[1] if wkey in msd else t.numel()
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            torch.nn.init.uniform_(t, -bound, bound)
    after = torch.norm(torch.cat([msd[k].flatten() for k in critic_keys])).item()

    if "optimizer_state_dict" in ckpt and "state" in ckpt["optimizer_state_dict"]:
        ckpt["optimizer_state_dict"]["state"] = {}

    torch.save(ckpt, args.dst)
    print(f"critic norm {before:.1f} -> {after:.1f}; optimizer state cleared; wrote {args.dst}")


if __name__ == "__main__":
    main()
