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

try:
    create_optimizer("prodigyplus", [nn.Parameter(torch.ones(2, 2))], 1e-4,
                     "fused_back_pass=True")
    fused_guard = False
except RuntimeError as e:
    fused_guard = "fused_back_pass=True" in str(e)
ck("Prodigy+ rejects unhooked fused_back_pass instead of making step() a no-op", fused_guard)

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

# Rotation replaces Parameter objects. Parameter state follows stable model names; adaptive
# group state follows the logical component/refiner cohort instead of leaking across windows.
with tempfile.TemporaryDirectory() as td:
    active_key = "blocks.0.mlp.fc1.weight"
    always_key = "token_refiner.proj.weight"
    p_active = nn.Parameter(torch.randn(8, 8))
    p_always = nn.Parameter(torch.randn(8, 8))
    old_groups = [
        {"params": [p_active], "lr_scale": 1.0, "fizgig_state_key": "window:fc1"},
        {"params": [p_always], "lr_scale": 1.0, "fizgig_state_key": "always"},
    ]
    old, _ = create_optimizer(
        "prodigyplus", old_groups, 1e-4, "stochastic_rounding=False")
    for _ in range(3):
        p_active.grad = torch.randn_like(p_active)
        p_always.grad = torch.randn_like(p_always)
        old.step()
        old.zero_grad(set_to_none=True)
    old.eval()

    # Make cohort identity observable independently of the stochastic training trajectory.
    old.param_groups[0]["d"], old.param_groups[0]["k"] = 1.23e-3, 17
    old.param_groups[1]["d"], old.param_groups[1]["k"] = 4.56e-4, 23
    eval_active = p_active.detach().clone()
    eval_always = p_always.detach().clone()

    store = RotatingOptimizerStateStore(td, fresh=True)
    store.stash([(active_key, p_active), (always_key, p_always)], old)
    store.mark_checkpoint("run-000003.safetensors", 3)
    ck("rotation store marks an exact checkpoint",
       store.matches_checkpoint("run-000003.safetensors", 3))

    p_active_new = nn.Parameter(eval_active.clone())
    p_always_new = nn.Parameter(eval_always.clone())
    exact_groups = [
        {"params": [p_active_new], "lr_scale": 1.0, "fizgig_state_key": "window:fc1"},
        {"params": [p_always_new], "lr_scale": 1.0, "fizgig_state_key": "always"},
    ]
    exact, _ = create_optimizer(
        "prodigyplus", exact_groups, 1e-4, "stochastic_rounding=False")
    restored = store.bind(
        [(active_key, p_active_new), (always_key, p_always_new)], exact)
    ck("fresh Parameters receive persisted optimizer state", restored == 2, restored)
    ck("both logical adaptive groups restore",
       store.last_group_restore_count == 2, store.last_group_restore_count)
    ck("component-specific Prodigy d survives its own rotation",
       float(exact.param_groups[0]["d"]) == 1.23e-3, exact.param_groups[0]["d"])
    ck("always-on Prodigy d restores independently",
       float(exact.param_groups[1]["d"]) == 4.56e-4, exact.param_groups[1]["d"])
    ck("component/refiner step counters remain independent",
       (int(exact.param_groups[0]["k"]), int(exact.param_groups[1]["k"])) == (17, 23),
       (exact.param_groups[0]["k"], exact.param_groups[1]["k"]))
    ck("Schedule-Free z survives Parameter replacement",
       "z" in exact.state[p_active_new] and "z" in exact.state[p_always_new])

    # A different component window must start its own d state, while the always-on refiner
    # continues from its persistent cohort.
    p_other = nn.Parameter(torch.randn(8, 8))
    p_always_again = nn.Parameter(eval_always.clone())
    other_groups = [
        {"params": [p_other], "lr_scale": 1.0, "fizgig_state_key": "window:fc2"},
        {"params": [p_always_again], "lr_scale": 1.0, "fizgig_state_key": "always"},
    ]
    other, _ = create_optimizer(
        "prodigyplus", other_groups, 1e-4, "stochastic_rounding=False")
    restored_other = store.bind(
        [("blocks.0.mlp.fc2.weight", p_other), (always_key, p_always_again)], other)
    ck("new component does not inherit another window's adaptive group state",
       float(other.param_groups[0]["d"]) == float(other.param_groups[0]["d0"])
       and int(other.param_groups[0]["k"]) == 1,
       (other.param_groups[0]["d"], other.param_groups[0]["k"]))
    ck("always-on cohort restores across a component change",
       float(other.param_groups[1]["d"]) == 4.56e-4
       and int(other.param_groups[1]["k"]) == 23,
       (other.param_groups[1]["d"], other.param_groups[1]["k"]))
    ck("only the matching always-on parameter state restores in a new component",
       restored_other == 1 and store.last_group_restore_count == 1,
       (restored_other, store.last_group_restore_count))

    # A normal rotation changes optimizer state and invalidates the old checkpoint marker.
    exact.train()
    exact.eval()
    store.stash([(active_key, p_active_new), (always_key, p_always_new)], exact)
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
