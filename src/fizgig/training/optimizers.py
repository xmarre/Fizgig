"""Optimizer selection, shared by the trainers.

Krea 2 hardcoded `bnb.optim.AdamW8bit` — a good default, but the only choice, which is the one
place the community comparison against OneTrainer (~40 optimizers) landed a fair hit.

Two things matter for a LoRA and pull against the "more optimizers is better" instinct:

* **Optimizer state is tiny here.** Krea 2 trains 264 small factors, so AdamW's two moments cost
  ~tens of MB against a 13-19 GB base. Choosing an 8-bit or factored optimizer to save memory is
  nearly pointless — the base dominates. What the choice actually buys is *update behaviour*.
* **Learning rates are not comparable across families.** Lion's update is a sign, so it wants
  roughly a tenth of AdamW's LR. Handing someone a dropdown without saying that is how you
  produce a fried LoRA and a bug report, so `create_optimizer` warns loudly on the record when
  the LR looks wrong for the family. Self-tuning families require trainer-level integration;
  Prodigy+ is exposed only where those semantics are implemented.

Free-form `module.path.ClassName` is accepted too, so a user who pip-installs something exotic
does not need a Fizgig release to use it.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import logging
import os
import shutil

import torch

logger = logging.getLogger(__name__)


# name -> (import to test, one-line description shown in the GUI/CLI help)
# Self-tuning families stay out of the generic Krea 2 dropdown by default because they own
# their learning-rate semantics and cannot be combined blindly with Fizgig's Adaptive LR.
# MiniMax opts into the supported subset explicitly. Prodigy+ is fail-closed: falling back to
# AdamW after switching the trainer to Prodigy's lr=1 semantics would be destructive.
_CATALOG = {
    "adamw8bit":          ("bitsandbytes", "AdamW, 8-bit state (default — the validated recipe)"),
    "adamw":              (None,           "AdamW, fp32 state, CUDA-fused where available"),
    "pagedadamw8bit":     ("bitsandbytes", "AdamW8bit that pages state to CPU under pressure"),
    "ademamix8bit":       ("bitsandbytes", "AdEMAMix — second slow EMA, aimed at long runs"),
    "pagedademamix8bit":  ("bitsandbytes", "AdEMAMix8bit with CPU paging"),
    "lion8bit":           ("bitsandbytes", "Lion — sign updates; use ~1/10 the AdamW LR"),
    "prodigyplus":         ("prodigyplus",  "Prodigy+ Schedule-Free — self-tuning LR; MiniMax opt-in"),
}

_SELF_TUNING = {"prodigyplus"}
_PRODIGY_PLUS_ALIASES = {
    "prodigyplus", "prodigy+", "prodigy-plus", "prodigy_plus",
    "prodigyplusschedulefree",
    "prodigyplus.prodigy_plus_schedulefree.prodigyplusschedulefree",
}

DEFAULT_OPTIMIZER = "adamw8bit"


def _canonical_name(name: str) -> str:
    key = (name or DEFAULT_OPTIMIZER).strip().lower()
    return "prodigyplus" if key in _PRODIGY_PLUS_ALIASES else key


def is_prodigy_plus(name: str) -> bool:
    return _canonical_name(name) == "prodigyplus"


def optimizer_uses_schedulefree(name: str, args_str: str = "") -> bool:
    if not is_prodigy_plus(name):
        return False
    return bool(parse_optimizer_args(args_str).get("use_schedulefree", True))


def prodigy_handles_gradient_scaling(args_str: str = "") -> bool:
    """Whether Prodigy+ already normalizes update scale internally.

    StableAdamW performs RMS update scaling when epsilon is numeric. With eps=None the
    optimizer switches to Adam-atan2, which is scale-invariant on its own. External gradient
    clipping is appropriate only when StableAdamW is disabled while retaining numeric epsilon.
    """
    kwargs = parse_optimizer_args(args_str)
    return kwargs.get("eps", 1e-8) is None or bool(kwargs.get("use_stableadamw", True))


def available_optimizers(include_self_tuning: bool = False) -> list[str]:
    """Catalog entries whose backing package is importable.

    Self-tuning optimizers are hidden unless a trainer explicitly opts in and implements their
    LR, clipping, save/eval, and resume semantics.
    """
    out = []
    for name, (module, _desc) in _CATALOG.items():
        if name in _SELF_TUNING and not include_self_tuning:
            continue
        if module is None:
            out.append(name)
            continue
        try:
            importlib.import_module(module)
            out.append(name)
        except Exception:
            pass
    return out


def describe(name: str) -> str:
    return _CATALOG.get(_canonical_name(name), (None, "custom optimizer"))[1]


def parse_optimizer_args(raw: str) -> dict:
    """`"weight_decay=0.01 betas=0.9,0.99"` -> `{"weight_decay": 0.01, "betas": (0.9, 0.99)}`.

    Values go through `ast.literal_eval`, so tuples, bools and None all survive; anything that
    isn't a literal is kept as a plain string rather than being executed.
    """
    kwargs = {}
    for tok in (raw or "").split():
        if "=" not in tok:
            raise ValueError(f"optimizer arg {tok!r} is not key=value")
        key, value = tok.split("=", 1)
        try:
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key] = value
    return kwargs


def _bnb(cls_name: str):
    import bitsandbytes as bnb
    return getattr(bnb.optim, cls_name)


def _warn_lr(name: str, lr: float) -> None:
    """A wrong-family LR is silent at step 1 and obvious only hours later. Say it now."""
    if name == "prodigyplus":
        return
    if name == "lion8bit" and lr > 5e-5:
        logger.warning("[optimizer] Lion applies the SIGN of the update, so it needs roughly a "
                       "TENTH of an AdamW LR. %.2e will likely overbake — try %.2e.", lr, lr / 10)
    elif name != "lion8bit" and lr > 1e-2:
        logger.warning("[optimizer] LR %.2e is very high for %s.", lr, name)


def create_optimizer(name: str, params, lr: float, args_str: str = "",
                     eps_floor_8bit: bool = False) -> tuple:
    """Build an optimizer. Returns `(optimizer, label)`; the label goes into LoRA metadata.

    Ordinary optimizers fall back to plain AdamW if construction fails. Self-tuning optimizers
    fail closed because the trainer may already have changed LR/clipping semantics for them.

    `eps_floor_8bit` raises the 8-bit Adam family's eps to 1e-6. OFF by default: it is a
    MiniMax H3 workaround and every other family keeps the library default. See the note below.
    """
    name = (name or DEFAULT_OPTIMIZER).strip()
    kwargs = parse_optimizer_args(args_str)
    key = _canonical_name(name)
    _warn_lr(key, lr)
    if key == "prodigyplus" and bool(kwargs.get("fused_back_pass", False)):
        raise RuntimeError(
            "[optimizer] Prodigy+ fused_back_pass=True requires trainer-installed "
            "post-accumulate hooks. Fizgig's MiniMax integration deliberately uses one "
            "coherent optimizer.step(), so this option is unsupported and must not silently "
            "turn step() into a no-op."
        )

    # eps 1e-6, not the library defaults' 1e-8 (matches ai-toolkit, which passes eps=1e-6 to
    # every Adam-family optimizer). This is a REAL stability bound, not a nicety: the 8-bit
    # optimizers store the second moment blockwise-quantized, and for heavily structured
    # gradients the small v entries quantize to ZERO — the update then degrades to lr*m/eps.
    # Measured on a MiniMax H3 epoch (46 steps @ 1e-4, eps=1e-8): lora_up drift reached 0.81
    # against an Adam bound of ~0.005 — the optimizer was applying ~100x the configured LR to
    # the most structured tensors (adaln worst, fc1 next), which presented as melted anatomy
    # at epoch 1. eps=1e-6 caps that amplification two orders of magnitude lower. Explicit
    # "eps=..." in Optimizer Args still wins.
    # NOTE the two conditions. 8-BIT Adam family only, NOT full-precision adam/adamw: full
    # precision has no quantized state, so v is whatever it really is, and a 1e-6 floor there
    # would DAMP the tensors with genuinely small second moments — the ones converging on fine
    # detail — while looking like a stability measure.
    #
    # And OPT-IN per caller, never global. This began as a MiniMax fix (bafb4e6) applied to the
    # whole Adam family, which silently moved Krea 2's DEFAULT optimizer off the library eps and
    # shipped that way in v3.3.0. Krea 2 never had the failure this works around and never asked
    # for the change. A workaround for one model family does not get to alter another's defaults;
    # the caller that needs it asks for it. Explicit "eps=..." in Optimizer Args still wins.
    if eps_floor_8bit and "8bit" in key and "lion" not in key:
        kwargs.setdefault("eps", 1e-6)

    try:
        if key == "prodigyplus":
            from prodigyplus import ProdigyPlusScheduleFree

            # Prodigy interprets lr as a multiplier on its learned d. Its documented default is
            # 1.0, so the ordinary Fizgig Learning Rate is not forwarded. Experts can override
            # the multiplier explicitly with optimizer_args="lr=...".
            prodigy_lr = float(kwargs.pop("lr", 1.0))
            if abs(float(lr) - prodigy_lr) > 1e-12:
                logger.info("[optimizer] Prodigy+ owns the step size: configured LR %.2e is "
                            "not used as its optimizer LR; multiplier=%.4g.", lr, prodigy_lr)

            # H3 can carry relative-LR groups. Preserve the dimensionless ratio, never the old
            # absolute Adam LR (e.g. 1e-4), which would starve Prodigy's d by four orders.
            pp_params = params
            if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
                pp_params = []
                for group in params:
                    g = dict(group)
                    g["lr"] = prodigy_lr * float(g.get("lr_scale", 1.0))
                    pp_params.append(g)
            opt = ProdigyPlusScheduleFree(pp_params, lr=prodigy_lr, **kwargs)
        elif key == "adamw8bit":
            opt = _bnb("AdamW8bit")(params, lr=lr, **kwargs)
        elif key == "pagedadamw8bit":
            opt = _bnb("PagedAdamW8bit")(params, lr=lr, **kwargs)
        elif key == "ademamix8bit":
            opt = _bnb("AdEMAMix8bit")(params, lr=lr, **kwargs)
        elif key == "pagedademamix8bit":
            opt = _bnb("PagedAdEMAMix8bit")(params, lr=lr, **kwargs)
        elif key == "lion8bit":
            opt = _bnb("Lion8bit")(params, lr=lr, **kwargs)
        elif key == "adamw":
            # Fused AdamW is one CUDA kernel over all 264 factors instead of a Python loop —
            # the "CUDA optimizer" the comparison thread was pointing at. Requires every param
            # on CUDA and floating point, which LoRA factors are.
            kwargs.setdefault("fused", torch.cuda.is_available())
            opt = torch.optim.AdamW(params, lr=lr, **kwargs)
        elif "." in name:
            module_path, cls_name = name.rsplit(".", 1)
            opt = getattr(importlib.import_module(module_path), cls_name)(params, lr=lr, **kwargs)
        else:
            raise ValueError(f"unknown optimizer {name!r} — use one of {available_optimizers()} "
                             "or a full module.path.ClassName")
    except Exception as e:
        if key == "prodigyplus":
            raise RuntimeError(
                f"[optimizer] could not create Prodigy+ ({e}). Refusing AdamW fallback because "
                "the trainer has already switched to Prodigy learning-rate semantics."
            ) from e
        logger.warning("[optimizer] could not create %s (%s) — falling back to AdamW", name, e)
        return torch.optim.AdamW(params, lr=lr), "adamw (fallback)"

    label = name + (f"({args_str.strip()})" if args_str.strip() else "")
    logger.info("optimizer: %s — %s", label, describe(key))
    return opt, label


def step_active_optimizer_groups(optimizer) -> bool:
    """Step only param groups that contain at least one gradient.

    Prodigy+ with split_groups=True advances group-level k/d/Schedule-Free statistics for every
    group present in optimizer.param_groups, even when every tensor in one group has grad=None.
    H3 rotation routing can intentionally produce exactly that shape: a component window may be
    fully frozen for one modality while the always-on refiner still trains. Temporarily excluding
    empty groups keeps no-op cohort optimizer clocks stationary, then restores the original
    group list before zero_grad/state/save code can observe it.
    """
    groups = optimizer.param_groups
    active = [group for group in groups
              if any(param.grad is not None for param in group.get("params", ()))]
    if not active:
        return False
    if len(active) == len(groups):
        optimizer.step()
        return True
    optimizer.param_groups = active
    try:
        optimizer.step()
    finally:
        optimizer.param_groups = groups
    return True


class RotatingOptimizerStateStore:
    """Disk-backed optimizer state for rotating Parameters with stable logical names.

    Parameter state is keyed by model tensor name. Param-group state is keyed separately by a
    trainer-supplied fizgig_state_key so Prodigy's adaptive d/step/Schedule-Free bookkeeping
    follows logical rotation cohorts instead of leaking from one component window into another.
    Inactive state lives on disk rather than pinning GPU or system RAM.
    """

    def __init__(self, root: str, *, fresh: bool = False):
        self.root = os.path.abspath(root)
        self._manifest_path = os.path.join(self.root, "manifest.json")
        self.last_group_restore_count = 0
        if fresh:
            shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.root, exist_ok=True)

    @staticmethod
    def _token(name: str) -> str:
        return hashlib.sha1(name.encode("utf-8")).hexdigest()[:20]

    def _state_path(self, name: str) -> str:
        return os.path.join(self.root, "param-" + self._token(name) + ".pt")

    def _group_path(self, key: str) -> str:
        return os.path.join(self.root, "group-" + self._token(key) + ".pt")

    @staticmethod
    def _group_key(group, index: int) -> str:
        key = group.get("fizgig_state_key")
        return str(key) if key is not None else f"group:{index}"

    @classmethod
    def _cpu_copy(cls, value):
        if torch.is_tensor(value):
            return value.detach().to("cpu").clone()
        if isinstance(value, dict):
            return {k: cls._cpu_copy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._cpu_copy(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._cpu_copy(v) for v in value)
        return value

    @classmethod
    def _device_copy(cls, value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {k: cls._device_copy(v, device) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._device_copy(v, device) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._device_copy(v, device) for v in value)
        return value

    def _write_manifest(self, payload: dict) -> None:
        tmp = self._manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
        os.replace(tmp, self._manifest_path)

    def _read_manifest(self) -> dict:
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError, TypeError):
            return {}

    def _has_group_state(self) -> bool:
        try:
            return any(name.startswith("group-") and name.endswith(".pt")
                       for name in os.listdir(self.root))
        except OSError:
            return False

    def stash(self, named_params, optimizer, *, preserve_checkpoint_marker: bool = False) -> None:
        """Persist live optimizer state by logical parameter and group identity."""
        if optimizer is None:
            raise RuntimeError("rotating optimizer state requires a live optimizer")

        # Validate the whole group layout before touching persistent state. Once mutation
        # begins, invalidate any checkpoint marker FIRST: a crash halfway through atomic
        # per-file replacements may leave a mixed sidecar, which must never still advertise
        # itself as an exact match for the previous checkpoint.
        group_keys = set()
        for index, group in enumerate(optimizer.param_groups):
            key = self._group_key(group, index)
            if key in group_keys:
                raise RuntimeError(f"duplicate rotating optimizer group key: {key!r}")
            group_keys.add(key)
        if not preserve_checkpoint_marker:
            self._write_manifest({})

        for index, group in enumerate(optimizer.param_groups):
            key = self._group_key(group, index)
            group_state = {
                k: self._cpu_copy(v)
                for k, v in group.items()
                if k not in ("params", "fizgig_state_key")
            }
            path = self._group_path(key)
            tmp = path + ".tmp"
            torch.save(group_state, tmp)
            os.replace(tmp, path)

        for name, param in named_params:
            state = optimizer.state.get(param)
            if not state:
                continue
            path = self._state_path(name)
            tmp = path + ".tmp"
            torch.save(self._cpu_copy(state), tmp)
            os.replace(tmp, path)

    def bind(self, named_params, optimizer) -> int:
        """Bind persisted logical state to freshly-created Parameter objects."""
        if optimizer is None:
            raise RuntimeError("rotating optimizer state requires a live optimizer")

        restored_groups = 0
        seen_group_keys = set()
        for index, group in enumerate(optimizer.param_groups):
            key = self._group_key(group, index)
            if key in seen_group_keys:
                raise RuntimeError(f"duplicate rotating optimizer group key: {key!r}")
            seen_group_keys.add(key)
            path = self._group_path(key)
            if not os.path.isfile(path):
                continue
            try:
                saved_group = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                saved_group = torch.load(path, map_location="cpu")
            device = group["params"][0].device if group.get("params") else "cpu"
            for field, value in saved_group.items():
                group[field] = self._device_copy(value, device)
            restored_groups += 1

        restored = 0
        for name, param in named_params:
            path = self._state_path(name)
            if not os.path.isfile(path):
                continue
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            optimizer.state[param] = self._device_copy(state, param.device)
            restored += 1

        if hasattr(optimizer, "parameters_to_process"):
            optimizer.parameters_to_process = None
        self.last_group_restore_count = restored_groups
        return restored

    def mark_checkpoint(self, checkpoint_path: str, epoch: int, state_id: str) -> None:
        if not state_id:
            raise ValueError("rotating optimizer checkpoint state_id must be non-empty")
        self._write_manifest({
            "checkpoint": os.path.basename(os.path.abspath(checkpoint_path)),
            "epoch": int(epoch),
            "state_id": str(state_id),
        })

    def matches_checkpoint(self, checkpoint_path: str, epoch: int, state_id: str) -> bool:
        if not state_id:
            return False
        manifest = self._read_manifest()
        return (
            manifest.get("checkpoint") == os.path.basename(os.path.abspath(checkpoint_path))
            and int(manifest.get("epoch", -1)) == int(epoch)
            and manifest.get("state_id") == str(state_id)
            and self._has_group_state()
        )

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
