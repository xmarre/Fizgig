"""Prodigy+ integration: construction, fail-closed semantics, rotation state and H3 CLI.

CPU-only. The real H3 full-finetune VRAM path remains a GPU integration test; these checks pin
the invariants that can regress without loading a 33B model.
"""
import inspect
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
import torch.nn as nn

from fizgig.minimax.rotation_ft import plan_h3_ft_windows
from fizgig.minimax.trainer import train_minimax
from fizgig.scripts.minimax_train import setup_parser
from fizgig.training.optimizers import (
    RotatingOptimizerStateStore,
    available_optimizers,
    create_optimizer,
    optimizer_uses_schedulefree,
)

fails = []


def ck(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{('  ' + str(detail)) if detail else ''}")
    if not cond:
        fails.append(label)


# Generic/Krea catalog stays unchanged; MiniMax explicitly opts into self-tuning optimizers.
ck("Prodigy+ stays out of the generic optimizer catalog",
   "prodigyplus" not in available_optimizers())
ck("MiniMax opt-in catalog contains Prodigy+",
   "prodigyplus" in available_optimizers(include_self_tuning=True))
ck("Prodigy+ defaults to Schedule-Free", optimizer_uses_schedulefree("prodigyplus", ""))
ck("use_schedulefree=False is respected",
   not optimizer_uses_schedulefree("prodigyplus", "use_schedulefree=False"))

# Fizgig's ordinary LR is NOT forwarded. Prodigy owns d and uses lr=1 as its multiplier.
p = nn.Parameter(torch.randn(8, 8))
opt, label = create_optimizer("prodigyplus", [p], 1e-4, "stochastic_rounding=False")
ck("Prodigy+ constructs", type(opt).__name__ == "ProdigyPlusScheduleFree", type(opt).__name__)
ck("configured Adam LR is not forwarded to Prodigy+", opt.param_groups[0]["lr"] == 1.0,
   opt.param_groups[0]["lr"])
before = p.detach().clone()
p.grad = torch.randn_like(p)
opt.step()
opt.zero_grad(set_to_none=True)
ck("Prodigy+ performs a real step", not torch.equal(before, p.detach()))

# Static depth-split ratios are dimensionless under Prodigy: 1.0 and 0.2, never 1e-4/2e-5.
p0 = nn.Parameter(torch.randn(4, 4))
p1 = nn.Parameter(torch.randn(4, 4))
groups = [
    {"params": [p0], "lr": 1e-4, "lr_scale": 1.0},
    {"params": [p1], "lr": 2e-5, "lr_scale": 0.2},
]
opt_groups, _ = create_optimizer(
    "prodigyplus", groups, 1e-4, "stochastic_rounding=False")
ck("Prodigy+ preserves relative param-group LR ratios",
   [g["lr"] for g in opt_groups.param_groups] == [1.0, 0.2],
   [g["lr"] for g in opt_groups.param_groups])

# A construction error must never silently become AdamW at lr=1 semantics.
try:
    create_optimizer("prodigyplus", [nn.Parameter(torch.ones(2, 2))], 1e-4,
                     "definitely_not_an_argument=True")
    fail_closed = False
except RuntimeError as e:
    fail_closed = "Refusing AdamW fallback" in str(e)
ck("Prodigy+ construction fails closed", fail_closed)

# Schedule-Free mode transitions are real and reversible at the API/state level.
p2 = nn.Parameter(torch.randn(8, 8))
opt2, _ = create_optimizer("prodigyplus", [p2], 1e-4, "stochastic_rounding=False")
p2.grad = torch.randn_like(p2)
opt2.step()
opt2.zero_grad(set_to_none=True)
opt2.eval()
ck("optimizer.eval() enters deploy mode", opt2.param_groups[0]["train_mode"] is False)
opt2.train()
ck("optimizer.train() returns to train mode", opt2.param_groups[0]["train_mode"] is True)

# Rotation replaces Parameter objects. State is restored by stable model key, including global d.
with tempfile.TemporaryDirectory() as td:
    p_old = nn.Parameter(torch.randn(8, 8))
    old, _ = create_optimizer("prodigyplus", [p_old], 1e-4, "stochastic_rounding=False")
    for _ in range(3):
        p_old.grad = torch.randn_like(p_old)
        old.step()
        old.zero_grad(set_to_none=True)
    old.eval()
    old_d = float(old.param_groups[0]["d"])
    old_k = int(old.param_groups[0]["k"])
    eval_weight = p_old.detach().clone()

    store = RotatingOptimizerStateStore(td, fresh=True)
    key = "blocks.0.mlp.fc1.weight"
    store.stash([(key, p_old)], old)
    store.mark_checkpoint("run-000003.safetensors", 3)
    ck("rotation store marks an exact checkpoint",
       store.matches_checkpoint("run-000003.safetensors", 3))

    p_new = nn.Parameter(eval_weight.clone())
    new, _ = create_optimizer("prodigyplus", [p_new], 1e-4, "stochastic_rounding=False")
    restored = store.bind([(key, p_new)], new)
    ck("fresh Parameter receives its persisted optimizer state", restored == 1, restored)
    ck("global Prodigy d survives rotation", float(new.param_groups[0]["d"]) == old_d,
       (new.param_groups[0]["d"], old_d))
    ck("global Prodigy step counter survives rotation", int(new.param_groups[0]["k"]) == old_k,
       (new.param_groups[0]["k"], old_k))
    ck("Schedule-Free z survives rotation", "z" in new.state[p_new])

    new.train()
    p_new.grad = torch.randn_like(p_new)
    new.step()
    ck("restored optimizer can continue stepping", int(new.param_groups[0]["k"]) == old_k + 1)

    # A normal rotation changes optimizer state and therefore invalidates the old checkpoint marker.
    new.eval()
    store.stash([(key, p_new)], new)
    ck("ordinary rotation invalidates an older checkpoint marker",
       not store.matches_checkpoint("run-000003.safetensors", 3))

# H3's parser and trainer signature expose both adapter and full-finetune Prodigy choices.
parser = setup_parser()
args = parser.parse_args([
    "--dit", "model.safetensors",
    "--dataset_config", "dataset.toml",
    "--output_dir", "out",
    "--output_name", "test",
    "--optimizer_type", "prodigyplus",
    "--finetune_optimizer_type", "prodigyplus",
])
ck("MiniMax CLI accepts Prodigy+ for LoRA/LoKR", args.optimizer_type == "prodigyplus")
ck("MiniMax CLI accepts Prodigy+ for full fine-tune",
   args.finetune_optimizer_type == "prodigyplus")
sig = inspect.signature(train_minimax)
ck("trainer exposes a separate full-finetune optimizer",
   "finetune_optimizer_type" in sig.parameters and "finetune_optimizer_args" in sig.parameters)

# Prodigy+ carries materially more live window state than fused Adafactor. The planner must
# become at least as conservative when the cost multiplier is applied.
base_windows, _, _ = plan_h3_ft_windows(32.0, n_blocks=50, window_cost_scale=1.0)
pp_windows, _, _ = plan_h3_ft_windows(32.0, n_blocks=50, window_cost_scale=3.5)
ck("Prodigy+ planner never chooses fewer/larger logical windows",
   pp_windows is None or (base_windows is not None and len(pp_windows) >= len(base_windows)),
   (None if base_windows is None else len(base_windows),
    None if pp_windows is None else len(pp_windows)))

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
sys.exit(1 if fails else 0)
