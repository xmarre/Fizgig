"""Rotating-window full fine-tune for MiniMax H3 — component windows on an NF4 base.

Subclasses krea2's BlockRotator (H3NF4Rotator), overriding exactly the quantization-coupled
methods. Component mode is THE H3 FT mode (24 Aug: the int8-residency block-window rotator
was removed — it never matched component's likeness speed): the base loads NF4-resident
(~10.5 GB, what lets one matmul across every block fit beside it), and activation is a
`__class__` swap from bnb Linear4bit to nn.Linear plus a weight swap from the bf16 master.

Two invariants this module exists to protect:

1. There is no raw bf16 file for H3 (the int8 checkpoint IS the deployed ground truth), so
   the CPU master is built by DEQUANTIZING the int8 file per tensor — exact with respect to
   the function the reference deploys. The NF4 residency is training-time only and never
   touches the master's fidelity.
2. Saves write bf16-master -> int8 ConvRot via save_full_checkpoint_h3, with STOCHASTIC
   rounding for trained stems (nearest is biased back to the base codes and silently erased
   sub-grid training deltas — the 24 Aug likeness-save bug). Untouched tensors copy through
   bit-exact from the source file.
"""

import hashlib
import json
import logging
import os
import shutil
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from fizgig.krea2.rotation import BlockRotator
from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen, stream_save_file
from fizgig.minimax.convrot import (dequantize_int8_convrot, parse_comfy_quant,
                                    quantize_int8_convrot)

logger = logging.getLogger(__name__)


class MasterStore:
    """Disk-backed bf16 master — the RAM dict's drop-in replacement for big fine-tunes.

    The in-RAM master (37.3 GB full-model) is only IRREPLACEABLE for tensors that have
    trained; an untouched entry is exactly reproducible by dequantizing the int8 source —
    and the access pattern is perfectly coarse (nothing between rotations, one window per
    boundary), so app-managed sequential spill beats letting Windows VM page 4K-at-random.
    Reads: scratch if the key has trained, else per-tensor dequant from the source (lazy —
    there is no build step at all). Writes: per-tensor raw .bin (uint16 view of the bf16
    bytes, np tofile — the house pattern; NO mmap, see the loader's Windows note) + a JSON
    manifest, both atomic via tmp + os.replace, superseded bytes deleted after. Spilling
    BF16 preserves the exactness invariant completely — bytes are bytes; an int8 master
    stays a non-starter (one lossy encode per rotation would compound across cycles).
    Duck-types the dict surface the rotators use: get / [] / in / keys."""

    def __init__(self, int8_path: str, scratch_dir: str, block_subset=None,
                 key_prefix: str = "blocks", include_prefixes=()):
        self.source_path = int8_path
        self.dir = scratch_dir
        os.makedirs(scratch_dir, exist_ok=True)
        self._f = MemoryEfficientSafeOpen(int8_path)
        keys = set(self._f.keys())
        allowed = (tuple(f"{key_prefix}.{int(b)}." for b in block_subset)
                   if block_subset is not None else None)
        self._quant: Dict[str, str] = {}     # master key -> quantized source stem
        self._dense: set = set()             # master keys stored dense in the source
        est_bytes = 0
        for k in sorted(keys):
            if not k.endswith(".comfy_quant"):
                continue
            stem = k[: -len(".comfy_quant")]
            if not stem.startswith(f"{key_prefix}."):
                continue
            if allowed is not None and not stem.startswith(allowed):
                continue
            self._quant[stem + ".weight"] = stem
            est_bytes += 2 * int(np.prod(self._f.header[stem + ".weight"]["shape"]))
        for pfx in include_prefixes:
            for k in sorted(keys):
                if (k.startswith(f"{pfx}.") and k.endswith(".weight")
                        and f"{k[:-len('.weight')]}.comfy_quant" not in keys
                        and k not in self._quant):
                    self._dense.add(k)
                    est_bytes += 2 * int(np.prod(self._f.header[k]["shape"]))
        if not self._quant:
            raise RuntimeError(
                f"MasterStore: no ConvRot tensors under '{key_prefix}.' in "
                f"{os.path.basename(int8_path)} — rotation FT needs the pre-quantized int8 "
                "checkpoint (the bf16 file has no comfy_quant markers and is not supported)")
        self.est_gb = est_bytes / 2 ** 30
        free = shutil.disk_usage(scratch_dir).free
        if free < est_bytes * 1.1:
            raise RuntimeError(
                f"[ft-master] the scratch drive has {free / 2**30:.1f} GB free but a fully "
                f"trained master spills ~{self.est_gb:.1f} GB — free space on "
                f"{os.path.splitdrive(scratch_dir)[0] or scratch_dir}, or use "
                f"--finetune_scratch_dir to point at a bigger fast drive, or "
                f"--finetune_master ram.")
        self._manifest_path = os.path.join(scratch_dir, "manifest.json")
        self._trained: Dict[str, dict] = {}  # key -> {"file", "shape"}
        if os.path.exists(self._manifest_path):
            try:
                with open(self._manifest_path, encoding="utf-8") as mf:
                    self._trained = json.load(mf).get("trained", {})
            except Exception:
                self._trained = {}
        logger.info("[ft-master] disk-backed master: %d tensors, ~%.1f GB when fully "
                    "trained, scratch at %s (RAM holds ~one tensor at a time)",
                    len(self._quant) + len(self._dense), self.est_gb, scratch_dir)

    # ---- dict surface (what the rotators and the save path actually call) --------------
    def keys(self):
        return list(self._quant) + sorted(self._dense)

    def __contains__(self, key) -> bool:
        return key in self._quant or key in self._dense

    def get(self, key, default=None):
        rec = self._trained.get(key)
        if rec is not None:
            arr = np.fromfile(os.path.join(self.dir, rec["file"]), dtype=np.uint16)
            return (torch.from_numpy(arr).view(torch.bfloat16)
                    .reshape(tuple(rec["shape"])))
        if key in self._quant:
            stem = self._quant[key]
            conf = parse_comfy_quant(self._f.get_tensor(stem + ".comfy_quant"))
            return dequantize_int8_convrot(self._f.get_tensor(stem + ".weight"),
                                           self._f.get_tensor(stem + ".weight_scale"),
                                           conf, out_dtype=torch.bfloat16)
        if key in self._dense:
            return self._f.get_tensor(key).to(torch.bfloat16)
        return default

    def __getitem__(self, key):
        t = self.get(key)
        if t is None:
            raise KeyError(key)
        return t

    def __setitem__(self, key, t: torch.Tensor):
        if key not in self:
            raise KeyError(f"MasterStore: {key} is not a master tensor")
        fn = hashlib.sha1(key.encode()).hexdigest()[:16] + ".bin"
        tmp = os.path.join(self.dir, fn + ".tmp")
        final = os.path.join(self.dir, fn)
        t.detach().to("cpu", dtype=torch.bfloat16).contiguous() \
            .view(torch.uint16).numpy().tofile(tmp)
        os.replace(tmp, final)               # the old version stays valid until this instant
        self._trained[key] = {"file": fn, "shape": list(t.shape)}
        mtmp = self._manifest_path + ".tmp"
        with open(mtmp, "w", encoding="utf-8") as mf:
            json.dump({"source": os.path.basename(self.source_path),
                       "trained": self._trained}, mf)
        os.replace(mtmp, self._manifest_path)

    def trained_keys(self):
        return set(self._trained)

    def close(self):
        try:
            self._f.file.close()
        except Exception:
            pass

    def cleanup(self):
        """Delete the scratch — call ONLY after the checkpoint that supersedes it is safely
        on disk (the scratch is the run's live training state until then)."""
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)


class _FlushedView:
    """master_state_dict for a MasterStore: live (active + always-on) tensors materialize
    from the GPU per key AT PRODUCTION TIME, everything else streams from the store — the
    base class's `dict(self.master)` would pull the whole master back into RAM at save."""

    def __init__(self, store, live):
        self._store, self._live = store, live

    def __getitem__(self, key):
        fn = self._live.get(key)
        return fn() if fn is not None else self._store[key]

    def __contains__(self, key):
        return key in self._live or key in self._store

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return set(self._store.keys()) | set(self._live)


def _h3_flushed_state_dict(rotator):
    """Shared master_state_dict for the H3 rotators: base behavior on a dict master, the
    lazy view on a MasterStore. Mirrors the base exactly — active window flushed from the
    live weights, always-on Linears exported unconditionally (dense ones were never in the
    master but HAVE trained)."""
    if not isinstance(rotator.master, MasterStore):
        return BlockRotator.master_state_dict(rotator)

    def _puller(lin):
        return lambda: lin.weight.detach().to("cpu", dtype=torch.bfloat16).clone()

    live = {}
    for key, lin in rotator._targets(list(rotator.active)):
        live[key] = _puller(lin)
    for prefix, module in rotator.always:
        for lname, lin in [(n, m) for n, m in module.named_modules()
                           if isinstance(m, nn.Linear)]:
            live[f"{prefix}.{lname}.weight"] = _puller(lin)
    return _FlushedView(rotator.master, live)


# Component windows for H3 (mode="component"): each window is ONE matmul of every block —
# full model depth per window, ~4 windows per cycle. Per-block bf16 GB at the pruned model's
# shapes (hidden 5376, ffn 14336, inner 5376): used by the trainer's defrag threshold.
H3_COMPONENT_PREFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
H3_COMPONENT_GB_PER_BLOCK = {"attn.qkv_proj": 0.174, "attn.out_proj": 0.058,
                             "mlp.fc1": 0.308, "mlp.fc2": 0.154}

# Component-FT VRAM model (DESIGN_component_ft_24gb.md). The design table's per-window
# peaks (qkv 21.6 / out 16.2 / fc1 24.7 / fc2 18.8) are GiB, and the first cut of this
# model subtracted them as if they were GB — a ~7% systematic underestimate that put the
# overhead at 13.3. Two independent field measurements on the 5090 (27 Aug, 496px stills,
# batch 1, fused backward, checkpointing) both put it at 14.3: a 25-block qkv window
# (4.35 GB) peaked 17.4 GiB, and a full-depth qkv window (8.7 GB) peaked 21.4 GiB —
# 22.98 - 8.7 = 14.28. Rounded UP to 14.5 so predictions err safe.
#
# Deliberately NOT modelled: the design note's "minus the active component's freed NF4
# share" term. Without it the fat windows over-predict (fc1/fc2 imply an overhead nearer
# 11-12), which over-splits them — conservative, never optimistic. Add it only with clip
# measurements in hand (see the full-model evaluation).
#
# CALIBRATED ON STILLS. A clip's activations are ~30x the tokens, so these do NOT
# transfer to video datasets; the trainer logs real per-window peaks every run.
# For clip datasets the caller subtracts ft_clip_activation_gb() from usable first.
_FT_OVERHEAD_GB = 14.5          # non-window peak with the full NF4 trunk resident
_FT_NF4_GB_PER_BLOCK = 0.21     # trunk share reclaimed per streamed block (10.5 / 50)
_FT_STREAM_SLOTS_GB = 2.0       # ring slots + copy-stream headroom when streaming
_FT_MAX_SANE_WINDOWS = 12       # past this, a full-speed cycle is slower than streaming

# The clip activation term (measured 28 Aug 2026, fizgig-ft-runs/clipq2 — six runs on a
# real Gizmo-spec clip at 0.25 MP): each latent frame beyond a still's single frame adds
# ~0.145 GB of per-step activation memory, linear across the measured range (7 -> 17
# latent frames, extrapolating cleanly to the 37-frame failures) and PLAN-INDEPENDENT —
# it is per-step memory, so splitting or streaming windows does not reduce it (the sim-24
# 124-frame run died in the FORWARD with small split windows). The stills overhead above
# already contains the 1-frame activation cost, hence (T - 1).
_FT_ACT_GB_PER_LATENT_FRAME = 0.145
# Fragmentation demands real headroom on clip runs: the 124-frame no-sim run fit on paper
# (28.4 predicted vs ~30.6 available) and still died — ~4 GiB sat reserved-but-unallocated
# with no expandable_segments on Windows. Applied only when clips are present so every
# field-proven stills plan stays bit-identical.
_FT_CLIP_FRAG_MARGIN_GB = 2.0


def ft_clip_activation_gb(latent_t, spatial_mp):
    """(activation_gb, margin_gb) a clip dataset costs on top of the stills overhead.

    latent_t is the LARGEST clip's latent frame count (grid T = 5n+2; a still is 1) and
    spatial_mp that same item's spatial megapixels — always the per-item pair, never
    max-T x max-mp across different items (per-step peaks belong to one item at a time;
    same two-maxima trap _max_effective_mp documents). Returned separately so console
    lines can name the two constants instead of conflating them. Zero for stills.

    mp scaling mirrors the LoRA planner's _ACT_GB_CKPT x mp/0.25 — modelled, only the
    0.25 MP anchor is measured. Batch size > 1 with clips is deliberately NOT modelled:
    both constants were measured at batch 1 (the FT norm), and silently scaling would
    dress an unmeasured extrapolation as a measurement."""
    t = max(1, int(latent_t))
    if t <= 1:
        return 0.0, 0.0
    act = _FT_ACT_GB_PER_LATENT_FRAME * (t - 1) * (float(spatial_mp) / 0.25)
    return act, _FT_CLIP_FRAG_MARGIN_GB


def plan_h3_ft_windows(usable_gb, subset=None, n_blocks=50, allow_stream=True,
                       window_cost_scale=1.0):
    """The H3 component-window plan for a VRAM budget: (windows, stream, reasons).

    A thin wrapper over the family-agnostic plan_component_windows with H3's calibrated
    constants — windows are bare prefixes where the full span fits, (prefix, lo, hi)
    depth-splits where it doesn't, and stream=True is the 16 GB tier (frozen
    out-of-window blocks ring in from CPU). window_cost_scale accounts for optimizer
    implementations whose live gradients/state scale with the active bf16 window; 1.0
    preserves the measured fused-Adafactor planner exactly. Returns (None, ..., reasons)
    when the budget can't run FT at all. Pure so the tier table is pinnable without a card."""
    from fizgig.krea2.rotation import plan_component_windows
    span = sorted(int(b) for b in subset) if subset else range(int(n_blocks))
    return plan_component_windows(
        usable_gb, span, n_blocks,
        {p: H3_COMPONENT_GB_PER_BLOCK[p] * float(window_cost_scale)
         for p in H3_COMPONENT_PREFIXES},
        overhead_gb=_FT_OVERHEAD_GB, trunk_gb_per_block=_FT_NF4_GB_PER_BLOCK,
        slots_gb=_FT_STREAM_SLOTS_GB, allow_stream=allow_stream,
        max_sane_windows=_FT_MAX_SANE_WINDOWS)


class H3NF4Rotator(BlockRotator):
    """BlockRotator for an NF4-resident H3 base (component-mode fine-tune).

    The NF4 residency (~10.5 GB vs int8's ~21) is what makes component windows fit 32 GB:
    one matmul across every block is up to 15.4 GB of bf16 (fc1). The trunk's ~9.5% NF4
    error is a training-time trade — the active window trains against a coarser frozen
    context than the deployed int8 base. The SAVED checkpoint is untouched by that trade:
    writes still go bf16-master -> int8 ConvRot via save_full_checkpoint_h3, so the output
    deploys exactly like the source checkpoint does.

    The loader builds NF4 targets as real bnb Linear4bit modules (meta-context shells,
    loader.py) — a CLASS with its own forward that asserts on a dense weight. So activation
    is the same `__class__` swap the ConvRot rotator does (Linear4bit inherits nn.Linear:
    the stock F.linear forward takes over, the bnb forward returns on restore), plus a
    weight-attribute swap. Re-encoding on deactivate must stage the weight through CPU:
    Params4bit only quantizes on the cpu->cuda move, and a cuda-born tensor would stay
    dense and trip bnb's `assert weight.shape[1] == 1` on the next forward (field crash).
    Discovery is a map built ONCE at construction — the Linear4bit class is only present
    while frozen, so the map must outlive activation."""

    def __init__(self, blocks: nn.ModuleList, master: Dict[str, torch.Tensor],
                 key_prefix: str = "blocks", device: str = "cuda", block_subset=None):
        super().__init__(blocks, master, key_prefix=key_prefix, device=device)
        self.touched: set = set()
        self.block_subset = set(int(b) for b in block_subset) if block_subset else None
        self._orig_class = {}            # id(linear) -> Linear4bit subclass, for restore
        # key -> (block_idx, lname, linear). Built while everything is still frozen.
        self._by_key: Dict[str, tuple] = {}
        for bi, block in enumerate(blocks):
            if self.block_subset is not None and bi not in self.block_subset:
                continue
            for lname, lin in block.named_modules():
                if type(lin).__name__ == "Linear4bit":
                    self._by_key[self._key(bi, lname)] = (bi, lname, lin)

    @staticmethod
    def _nf4(w: torch.Tensor, device) -> nn.Parameter:
        """Re-encode a bf16 weight exactly as the loader does. The CPU staging is load-
        bearing: Params4bit quantizes on the cpu->cuda transition only."""
        from bitsandbytes.nn import Params4bit
        return Params4bit(w.detach().to("cpu", dtype=torch.bfloat16),
                          requires_grad=False, quant_type="nf4").to(device)

    def master_state_dict(self):
        return _h3_flushed_state_dict(self)

    def _targets(self, spec) -> List[tuple]:
        from fizgig.krea2.rotation import is_component_spec, component_entry_matches
        if is_component_spec(spec):
            return [(k, lin) for k, (bi, lname, lin) in self._by_key.items()
                    if any(component_entry_matches(c, lname, bi) for c in spec)]
        want = set(int(b) for b in spec)
        return [(k, lin) for k, (bi, lname, lin) in self._by_key.items() if bi in want]

    def _activate_targets(self, targets) -> int:
        n = 0
        for key, lin in targets:
            w = self.master.get(key)
            if w is None:
                logger.warning("[h3-nf4] no master weight for %s — leaving frozen", key)
                continue
            self._orig_class[id(lin)] = type(lin)
            lin.__class__ = nn.Linear      # the bnb 4-bit forward vanishes with the class
            lin.weight = nn.Parameter(w.to(self.device, dtype=torch.bfloat16),
                                      requires_grad=True)
            self.touched.add(key)
            n += 1
        return n

    def _deactivate_targets(self, targets) -> int:
        n = 0
        for key, lin in targets:
            if key not in self.master:
                continue
            trained = lin.weight.detach()
            self.master[key] = trained.to("cpu", dtype=torch.bfloat16).clone()
            lin.weight = self._nf4(trained, self.device)
            lin.__class__ = self._orig_class.pop(id(lin))
            # RELEASE THE ORPHAN. Rebinding lin.weight is not enough: something C++-side
            # (autograd's AccumulateGrad, invisible to gc.get_referrers) keeps the old
            # bf16 storage alive, so the whole outgoing window survived into the NEXT
            # epoch — measured at the boundary that failed full-model FT on a 24 GB
            # budget (22.03 GiB LIVE of a 22.24 GiB cap, only 207 MiB stranded, i.e. not
            # fragmentation). Neither gc nor the park/restore defrag can reach it, because
            # by now it is no longer a module parameter for the walk to find.
            # resize_(0) frees the bytes under every tensor sharing the storage and
            # leaves phantom holders a zero-byte husk — the same trick, and the same
            # justification, as park_dit_partial. Safe here because the value is already
            # saved to the master above and _nf4 re-encodes from its own CPU copy, so
            # nothing legitimate reads this storage again.
            # GUARDED, because "frees the bytes under every tensor sharing the storage"
            # cuts both ways: if a re-encode ever returned a VIEW of the trained weight
            # rather than its own copy, this would hand the module a zero-byte weight and
            # the next forward would die on a size-0 storage. Real bnb packs to fresh
            # uint8 so it cannot alias, but that is the encoder's property, not ours —
            # assert it here rather than trusting every future one to keep it.
            _orphan = trained.untyped_storage()
            del trained
            if lin.weight.untyped_storage().data_ptr() != _orphan.data_ptr():
                try:
                    _orphan.resize_(0)
                except Exception:
                    pass
            n += 1
        return n

    def activate_always(self, prefix: str, module: nn.Module) -> int:
        """Token refiner (+ condition_proj) under NF4 residency: the refiner's big Linears
        load as Linear4bit too (the loader's NF4 targets include token_refiner.blocks), so
        they take the same class-swap; small dense Linears just unfreeze. Master entries
        for the refiner come from build_bf16_master_h3's include_prefixes — dense in the
        source file, no dequant involved."""
        n_swapped = n_direct = 0
        for lname, lin in [(n, m) for n, m in module.named_modules()
                           if isinstance(m, nn.Linear)]:
            key = f"{prefix}.{lname}.weight"
            if type(lin).__name__ == "Linear4bit":
                w = self.master.get(key)
                if w is None:
                    logger.warning("[h3-nf4] no master weight for always-on %s — leaving "
                                   "frozen", key)
                    continue
                self._orig_class[id(lin)] = type(lin)
                lin.__class__ = nn.Linear
                lin.weight = nn.Parameter(w.to(self.device, dtype=torch.bfloat16),
                                          requires_grad=True)
                self.touched.add(key)
                n_swapped += 1
            else:
                lin.weight.requires_grad_(True)
                self.touched.add(key)
                n_direct += 1
            if lin.bias is not None:
                lin.bias.requires_grad_(True)
        self.always.append((prefix, module))
        logger.info("[h3-nf4] always-on: %s (%d Linears trainable for the whole run%s)",
                    prefix, n_swapped + n_direct,
                    f"; {n_swapped} de-quantized, {n_direct} already dense" if n_swapped
                    else "")
        return n_swapped + n_direct


def build_bf16_master_h3(int8_path: str, block_subset=None,
                         key_prefix: str = "blocks",
                         include_prefixes=()) -> Dict[str, torch.Tensor]:
    """The CPU bf16 master for an H3 rotation FT, dequantized from the int8 checkpoint.

    Streams tensor-by-tensor (never holds the file whole). block_subset limits the master to
    a --finetune_blocks selection — ~0.77 GB per block, so 20-49 costs ~23 GB instead of the
    full model's ~38.6. Keys match H3NF4Rotator._key(): '<prefix>.<i>.<lname>.weight'.
    include_prefixes adds DENSE tensors verbatim (the token refiner for the NF4 rotator's
    always-on path — unquantized in the source file, so no dequant, just a bf16 copy)."""
    allowed = None
    if block_subset is not None:
        allowed = tuple(f"{key_prefix}.{int(b)}." for b in block_subset)
    master: Dict[str, torch.Tensor] = {}
    n = 0
    with MemoryEfficientSafeOpen(int8_path) as f:
        keys = set(f.keys())
        for k in sorted(keys):
            if not k.endswith(".comfy_quant"):
                continue
            stem = k[: -len(".comfy_quant")]
            if not stem.startswith(f"{key_prefix}."):
                continue
            if allowed is not None and not stem.startswith(allowed):
                continue
            conf = parse_comfy_quant(f.get_tensor(k))
            dense = dequantize_int8_convrot(f.get_tensor(stem + ".weight"),
                                            f.get_tensor(stem + ".weight_scale"),
                                            conf, out_dtype=torch.bfloat16)
            master[stem + ".weight"] = dense.cpu()
            n += 1
        for pfx in include_prefixes:
            for k in sorted(keys):
                if (k.startswith(f"{pfx}.") and k.endswith(".weight")
                        and k not in master and f"{k[:-len('.weight')]}.comfy_quant" not in keys):
                    master[k] = f.get_tensor(k).to(torch.bfloat16).cpu()
                    n += 1
    gb = sum(t.numel() * t.element_size() for t in master.values()) / 2 ** 30
    logger.info("[h3-rotation] bf16 master built: %d tensors, %.1f GB CPU RAM "
                "(dequantized from the int8 checkpoint — the deployed ground truth)", n, gb)
    if n == 0:
        raise RuntimeError(
            f"build_bf16_master_h3: no ConvRot tensors under '{key_prefix}.' in "
            f"{os.path.basename(int8_path)} — rotation FT needs the pre-quantized int8 "
            "checkpoint (the bf16 file has no comfy_quant markers and is not supported)")
    return master


def save_full_checkpoint_h3(rotator: "H3NF4Rotator", src_path: str, out_path: str,
                            extra_metadata: Dict[str, str] = None) -> str:
    """Write a full int8-ConvRot checkpoint: the source file with trained tensors overlaid.

    Trained ConvRot stems are re-encoded (quantize_int8_convrot) as they stream to disk —
    weight int8 + weight_scale reshaped to the SOURCE tensor's exact shape + comfy_quant
    copied verbatim — so the output loads everywhere the source does (ComfyUI, the loader,
    diff-to-LoRA). Trained dense tensors (the always-on refiner) overlay in the source
    dtype. Every untouched tensor copies through bit-exact. One tensor in memory at a time
    (stream_save_file); atomic via tmp + os.replace."""
    # Announced up front: this streams a ~21 GB file and the progress bar sits still the
    # whole time — without a line here the run looks hung at every cycle boundary (field).
    logger.info("[h3-ft] saving full checkpoint -> %s (~%.0f GB) — this can take a few "
                "minutes, training resumes when it's done...",
                os.path.basename(out_path),
                os.path.getsize(src_path) / 2 ** 30 if os.path.isfile(src_path) else 21)
    flushed = rotator.master_state_dict()
    touched_stems = {k[: -len(".weight")] for k in rotator.touched if k.endswith(".weight")}

    with MemoryEfficientSafeOpen(src_path) as src:
        src_meta = dict(src.metadata() or {})
        header = {k: src.header[k] for k in src.keys()}
        rot_by_stem = {}
        for stem in touched_stems:
            ck = stem + ".comfy_quant"
            if ck in header:
                conf = parse_comfy_quant(src.get_tensor(ck))
                rot_by_stem[stem] = int(conf.get("convrot_groupsize", 256)) if conf.get("convrot") else 1

    meta = dict(src_meta)
    meta.update({"fizgig_finetune": "minimax-h3-rotation",
                 "fizgig_trained_tensors": str(len(rotator.touched)),
                 "fizgig_source_base": os.path.basename(src_path)})
    if extra_metadata:
        meta.update({str(k): str(v) for k, v in extra_metadata.items()})

    _DT = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
           "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
           "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}

    # Re-encode each trained stem ONCE: the weight producer computes (q, s) and parks the
    # scale for the weight_scale producer (source key order is not guaranteed). Scales are
    # [out, 1]-tiny, so a parked handful costs nothing. One reader handle serves every
    # untouched-tensor copy — the alternative (an open per tensor) re-parses the ~21 GB
    # file's header nine hundred times.
    _pending_scales: Dict[str, torch.Tensor] = {}
    _reader = {"f": None}

    def _encode(stem):
        # Stochastic rounding for TRAINED stems — nearest is biased back to the base codes,
        # so sub-grid training deltas (the 0.2-1% an FT actually produces) rounded to
        # near-zero training signal in the file while previews (which render the exact
        # master) showed full likeness (field, 24 Aug — the checkpoint-vs-preview mystery).
        # Deterministic per-stem seed so re-saving the same master gives the same bytes.
        gen = torch.Generator()
        gen.manual_seed(int.from_bytes(hashlib.sha1(stem.encode()).digest()[:8], "big")
                        & 0x7FFFFFFFFFFFFFFF)
        return quantize_int8_convrot(flushed[stem + ".weight"], rot_by_stem.get(stem, 256),
                                     stochastic=True, generator=gen)

    def make_producer(key):
        info = header[key]
        dt, shape = _DT[info["dtype"]], tuple(info["shape"])
        stem = None
        if key.endswith(".weight") and key[: -len(".weight")] in touched_stems:
            stem = key[: -len(".weight")]
            kind = "weight"
        elif key.endswith(".weight_scale") and key[: -len(".weight_scale")] in touched_stems:
            stem = key[: -len(".weight_scale")]
            kind = "scale"
        if stem is not None and dt == torch.int8 and kind == "weight":
            def _p(stem=stem, shape=shape):
                q, s = _encode(stem)
                _pending_scales[stem] = s
                return q.reshape(shape)
            return dt, shape, _p
        if stem is not None and kind == "scale":
            def _p(stem=stem, dt=dt, shape=shape):
                s = _pending_scales.pop(stem, None)
                if s is None:
                    s = _encode(stem)[1]
                return s.to(dt).reshape(shape)
            return dt, shape, _p
        if stem is not None and kind == "weight":
            # a touched DENSE tensor (always-on refiner): overlay in the source dtype
            def _p(key=key, dt=dt, shape=shape):
                return flushed[key].to(dt).reshape(shape)
            return dt, shape, _p

        def _p(key=key):
            return _reader["f"].get_tensor(key)
        return dt, shape, _p

    specs = {k: make_producer(k) for k in header}
    tmp = out_path + ".tmp"
    try:
        _reader["f"] = MemoryEfficientSafeOpen(src_path)
        try:
            stream_save_file(specs, tmp, metadata=meta)
        finally:
            _reader["f"].file.close()
        os.replace(tmp, out_path)
    except BaseException:
        # KeyboardInterrupt/SystemExit are exactly the truncation cases — never leave a
        # half-written file that LOOKS like a checkpoint.
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    logger.info("[h3-rotation] saved full checkpoint -> %s (%d trained tensors overlaid)",
                out_path, len(rotator.touched))
    return out_path
