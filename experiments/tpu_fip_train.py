"""FIP-Net on Kaggle TPU (v5e-8, torch_xla 2.8 / PJRT).

Backend + training loop, mirroring the proven GiantLM TPU patterns:

  * single-core path: in-kernel device probe + sanity (run_tpu.py style)
  * multi-core: xmp.spawn from a PRISTINE process (the launcher never touches
    a device on that path; the device count probe runs in a throwaway child)
  * Kaggle's stale pod-slice env (TPU_PROCESS_ADDRESSES / CLOUD_TPU_TASK_ID)
    is dropped spawn-only so libtpu re-derives the true 8-chip topology
  * per worker: fwd/bwd, manual grad all_reduce (SUM, 1/ws), opt.step(),
    mark_step() once per optimizer step, and its own DISJOINT shard of
    window starts (deterministic stride of one shared permutation)
  * master only (global_ordinal() == 0 - NOT xm.is_master_ordinal(), which is
    per-HOST and true on every core of a pod slice): eval, ckpt, JSONL log
  * PJRT replicates: --batch is the GLOBAL batch (Kaggle guidance ~128/core,
    i.e. --batch 1024 on 8 cores) - unlike ddp_train.py where --batch is
    per rank
  * XLA specifics: no CUDA graphs (plain eager step + mark_step), no
    torch.compile, constant LR (a python float changes the traced graph per
    step - recompile storm; a device tensor schedule needs more plumbing),
    bf16 cross_entropy means TPU-vs-TPU comparisons only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class SingleCore:
    """One XLA device, no collectives."""
    ordinal = 0
    ws = 1

    def __init__(self):
        import torch_xla.core.xla_model as xm
        self.xm = xm
        self.device = xm.xla_device()

    def mark_step(self):
        self.xm.mark_step()

    def all_reduce_grads(self, params):
        pass  # replicated single core

    def is_master(self):
        return True


class MpCore:
    """One core of N (xmp.spawn); PJRT replicates the model."""
    ordinal = None
    ws = None

    def __init__(self):
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        self.xm = xm
        self.device = xm.xla_device()
        self.ws = xr.world_size()
        self.ordinal = xr.global_ordinal()

    def mark_step(self):
        self.xm.mark_step()

    def all_reduce_grads(self, params):
        grads = [p.grad for p in params if p.grad is not None]
        if grads:
            self.xm.all_reduce(self.xm.REDUCE_SUM, grads, scale=1.0 / self.ws)

    def is_master(self):
        # global_ordinal, not xm.is_master_ordinal (per-host)
        return self.ordinal == 0


def shard_starts(n_total, ordinal, ws, seed=0):
    """Disjoint window starts: one shared permutation, strided by ordinal.
    Without it every core trains on identical windows (8x wasted work)."""
    g = np.random.default_rng(seed)
    perm = g.permutation(n_total)
    return perm[ordinal::ws]


def write_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def save_ck(path, core, m, opt, step, args, best, traj):
    if not core.is_master():
        return
    tmp = path + ".tmp"
    torch_save({"model": _unwrap(m).state_dict(), "opt": opt.state_dict(),
                "step": step, "best": best, "traj": traj, "args": vars(args)},
               tmp)
    os.replace(tmp, path)


def torch_save(obj, path):
    import torch
    torch.save(obj, path)


def _unwrap(m):
    return m


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="data/train_tokens.npy")
    ap.add_argument("--val", default="data/val_tokens.npy")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--bytes-mat", default="data/bytes_mat.pt")
    ap.add_argument("--codebook", default="data/codebook_wiki8192.pt",
                    help="precomputed on CPU by the prep cell; workers only "
                         "load it (8x optimize_codebook would be waste)")
    ap.add_argument("--n-bytes", type=int, default=12)
    ap.add_argument("--d-code", type=int, default=32)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--dim", type=int, default=640)
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--n-attn", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--block-kind", default="gated",
                    choices=["gated", "half", "rot"])
    ap.add_argument("--iso-lambda", type=float, default=1e-3)
    ap.add_argument("--attn-mode", default="half", choices=["replace", "half"],
                    help="half = (h+attn)/sqrt2 identity preservation - required "
                         "for deep (17-layer) attention stacks")
    ap.add_argument("--batch", type=int, default=1024,
                    help="GLOBAL batch (PJRT replicates); ~128/core")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="constant on TPU (see module docstring)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear LR ramp over N steps (recompiles once)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-windows", type=int, default=2048)
    ap.add_argument("--save", default="/kaggle/working/ck_tpu_fip.pt")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log", default="/kaggle/working/runs/fip_tpu.jsonl")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--smoke-xla", action="store_true")
    ap.add_argument("--tpu-parallel", action="store_true")
    return ap


def build_model(a):
    import torch
    import experiments.attention_v3 as av
    bm = torch.load(a.bytes_mat, map_location="cpu") if a.bytes_mat else None
    m = av.AttLM(input_mode="byte", train_allpos=True, pos_mode="rope",
                 block_kind=a.block_kind, dim=a.dim, heads=a.heads,
                 n_attn=a.n_attn, depth_after=a.depth, code_dim=a.d_code,
                 n_bytes=a.n_bytes, d_code=a.d_code, vocab=a.vocab,
                 window=a.window, bytes_mat=bm,
                 attn_mode=getattr(a, "attn_mode", "replace"))
    if a.iso_lambda:
        m.iso_lambda = a.iso_lambda
    m.decoder.codes(cache_path=a.codebook)
    return m


def eval_chunked(core, m, vx, vy, chunk=128):
    """V=8192 allpos logits for 2048 windows = 4.3GB bf16/fp32 - chunk it."""
    import torch
    preds, vlosses = [], []
    with torch.no_grad():
        for i in range(0, vx.shape[0], chunk):
            xs, ys = vx[i:i + chunk], vy[i:i + chunk]
            preds.append(m.predict(xs))
            vlosses.append(m.loss(xs, ys).item() * xs.shape[0])
    core.mark_step()
    pred = torch.cat(preds)
    acc = (pred == vy).float().mean().item()
    acc_l = (pred[:, -1] == vy[:, -1]).float().mean().item()
    return acc, acc_l, sum(vlosses) / vx.shape[0]


def run(core, a):
    import torch
    device = core.device

    fit = np.load(a.fit, mmap_mode="r")
    val = np.load(a.val, mmap_mode="r")
    # this worker's disjoint shard of window starts
    starts = shard_starts(len(fit) - a.window - 1, core.ordinal, core.ws)
    starts = torch.from_numpy(starts[: min(len(starts), 12_500_000)]).to(device)
    val_t = torch.from_numpy(np.array(val[:2_000_000])).to(device).to(torch.int32)

    W, B = a.window, a.batch
    def sample(data, count):
        pos = data[torch.randint(0, data.shape[0], (count,), device=device)]
        off = torch.arange(W, device=device)
        idx = pos.unsqueeze(1) + off.unsqueeze(0)
        return data[idx].long(), data[idx + 1].long()

    torch.manual_seed(0)                     # identical init on all cores
    m = build_model(a).to(device)
    params = sum(p.numel() for p in m.parameters())
    if core.is_master():
        print(f"model: dim {a.dim} depth {a.depth} n_attn {a.n_attn} "
              f"heads {a.heads} | {params/1e6:.2f}M params | "
              f"ws {core.ws} global batch {B}", flush=True)
    core.mark_step()

    vx, vy = sample(val_t, a.eval_windows)   # same val windows on every core

    opt = torch.optim.Adam(m.parameters(), lr=a.lr, betas=(0.8, 0.999))
    step0, best, traj = 1, 0.0, []
    last_path = a.save + ".last"
    for p in (last_path, a.save):
        if os.path.exists(p):
            try:
                st = torch.load(p, map_location="cpu")
                m.load_state_dict(st["model"])
                opt.load_state_dict(st["opt"])
                step0, best, traj = st["step"] + 1, st["best"], st["traj"]
                if core.is_master():
                    print(f"resumed at step {st['step']} from {p}", flush=True)
                break
            except (RuntimeError, OSError, EOFError, KeyError):
                pass

    t0 = time.time()
    dt = 0.0
    for s in range(step0, a.steps + 1):
        ts = time.perf_counter()
        x, y = sample(starts, B)
        loss = m.loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        core.all_reduce_grads(list(m.parameters()))
        opt.step()
        core.mark_step()                     # once per optimizer step
        dt = (time.perf_counter() - ts) * 1000
        if core.is_master() and a.log_every and s % a.log_every == 0:
            print(f"  step {s:7d}  train {float(loss.detach()):.4f}  {dt:6.0f}ms"
                  f"  eta {(a.steps - s) * max(dt, 1) / 1000 / 60:5.1f}min",
                  flush=True)
        if s % a.eval_every == 0 or s == a.steps:
            acc, acc_l, vloss = eval_chunked(core, m, vx, vy)
            tag = ""
            if acc > best:
                best, tag = acc, "  *best*"
                save_ck(a.save, core, m, opt, s, a, best, traj)
            traj.append(f"{s//1000}k:{acc*100:.1f}")
            if core.is_master():
                print(f"  step {s:7d}  val {vloss:.3f}  acc(all) {acc*100:6.2f}%"
                      f"  acc(last) {acc_l*100:6.2f}%{tag}  {dt:.0f}ms", flush=True)
                write_jsonl(a.log, {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "eval",
                    "step": s, "val_loss": vloss, "acc_all": acc,
                    "acc_last": acc_l, "is_best": bool(tag),
                    "train_loss": float(loss.detach()), "ms": dt, "ws": core.ws,
                })
        if s % a.save_every == 0:
            save_ck(last_path, core, m, opt, s, a, best, traj)
    save_ck(last_path, core, m, opt, a.steps, a, best, traj)
    if core.is_master():
        print(f"FIP-TPU[{a.dim}x{a.depth},na{a.n_attn}] ws{core.ws} {a.steps}: "
              f"BEST {best*100:.2f}% [{' '.join(traj)}]", flush=True)
    return 0


def _mp_fn(index):
    a = _MP_ARGS
    core = MpCore()
    run(core, a)


_MP_ARGS = None


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)

    if a.tpu_parallel:
        # xmp.spawn requires a PRISTINE process: never touched an XLA device,
        # never imported the runtime for real. The count probe runs in a
        # throwaway child precisely to keep this parent pristine.
        import torch_xla.distributed.xla_multiprocessing as xmp
        for k in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
            os.environ.pop(k, None)          # stale pod-slice view (spawn-only)
        global _MP_ARGS
        _MP_ARGS = a                          # Namespace survives pickling
        xmp.spawn(_mp_fn, args=(), nprocs=None)
        return 0

    # single-core: in-kernel probe is authoritative here
    import torch
    import torch_xla
    print(f"torch {torch.__version__} / torch_xla {torch_xla.__version__} "
          f"(match={torch.__version__.split('+')[0] == torch_xla.__version__})",
          flush=True)
    print(f"PJRT_DEVICE={os.environ.get('PJRT_DEVICE')!r}", flush=True)
    core = SingleCore()
    a2 = torch.tensor(100, device=core.device)
    b2 = torch.tensor(200, device=core.device)
    total = int((a2 + b2).item())
    print(f"  XLA sanity: 100 + 200 = {total} on {core.device}", flush=True)
    if total != 300:
        return 1
    if a.smoke_xla:
        print(f"OK: sanity passed on {core.device}", flush=True)
        return 0
    for f in (a.fit, a.val, a.bytes_mat):
        if not os.path.exists(f):
            return _missing(f)
    return run(core, a)


def _missing(f):
    print(f"missing: {f}\ncwd is {os.getcwd()}. Prepare data first "
          f"(prepare_wikitext.py + the codebook prep cell) or attach a "
          f"Kaggle dataset.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
