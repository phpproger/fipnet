"""DDP training for AttLM (FIP resonance LM) - Kaggle T4 x 2 ready.

Launch via torchrun (Kaggle/cluster):
  torchrun --standalone --nproc_per_node=2 experiments/ddp_train.py \
      --dim 640 --depth 14 --n-attn 2 ...

Launch locally (single GPU, no torchrun):
  python experiments/ddp_train.py --dim 320 ...

Design notes (mirroring the proven GiantLM ddp_train.py):
  * Rendezvous: torchrun env vars (MASTER_ADDR/MASTER_PORT via its own
    TCPStore). NEVER a file:// store - rank 0 deleting the store file while
    rank 1 still holds the inode makes the ranks never meet.
  * Backend: gloo by default (works on Windows and in Kaggle containers);
    opt into NCCL with FIP_BACKEND=nccl.
  * Progress: cheap per-step log with ETA (--log-every) plus a JSONL eval
    log flushed per record - the only signal that survives `!torchrun`
    stdout buffering in notebooks.
  * Checkpoints: rank 0 only, atomic (tmp + os.replace), saved WITHOUT the
    DDP "module." prefix. `--ckpt` is the BEST checkpoint (only overwritten
    on improvement); `--ckpt.last` is the periodic resume point.
  * Divergence guard: at eval, a checksum of one parameter is all-gathered
    across ranks; a spread turns the silent no-reduction failure into a
    loud one.
  * Data: each rank samples windows from its disjoint segment - no sampler
    sync, allpos targets stay local.
  * Memory hygiene: torch.cuda.empty_cache() after every eval.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiments.attention_v3 as av


# --------------------------------------------------------------------------- #
# distributed setup
# --------------------------------------------------------------------------- #
def setup_distributed(rank=None, world=None):
    """Returns (rank, world, local, device).

    torchrun path (rank is None): env:// rendezvous - MASTER_ADDR/PORT set
    by torchrun's own TCPStore; NO file:// store anywhere near it.
    Local fallback (rank given): FileStore, portable (Windows quirk-free).
    """
    torchrun_mode = rank is None
    if torchrun_mode:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
    if world > 1:
        backend = os.environ.get("FIP_BACKEND", "gloo")
        if os.name != "nt":                       # container NCCL hygiene
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        else:
            backend = "gloo"                      # Windows has no nccl
        if not dist.is_initialized():
            if torchrun_mode:
                dist.init_process_group(backend=backend, rank=rank,
                                        world_size=world)
            else:
                store_path = os.path.abspath(".fip_ddp_store")
                if rank == 0 and os.path.exists(store_path):
                    os.remove(store_path)
                dist.init_process_group("gloo",
                                        init_method=f"file://{store_path}",
                                        rank=rank, world_size=world)
    if torch.cuda.is_available():
        local = int(os.environ.get("LOCAL_RANK", rank if not torchrun_mode else 0))
        n_gpu = torch.cuda.device_count()
        if n_gpu:
            local = local % n_gpu
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
        else:
            device = torch.device("cpu")
    else:
        local = 0
        device = torch.device("cpu")
    return rank, world, local, device


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, *args) -> None:
    if is_main(rank):
        print(*args, flush=True)


def write_jsonl(rank: int, path: str, record: dict) -> None:
    """Append one JSON line and flush. The progress signal that survives
    `!torchrun` output buffering and a cut session."""
    if not is_main(rank):
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def save_ck(rank: int, path: str, core: nn.Module, opt, step: int, world: int,
            args, best: float, traj: list) -> None:
    """Rank 0 only, atomic. Carries optimizer + bookkeeping so a resume
    restarts exactly where it left off."""
    if not is_main(rank):
        return
    tmp = path + ".tmp"
    torch.save({"model": core.state_dict(), "opt": opt.state_dict(),
                "step": step, "world": world, "best": best, "traj": traj,
                "args": vars(args)}, tmp)
    os.replace(tmp, path)


class LossModule(nn.Module):
    """Wraps model.loss so DDP hooks see every parameter in the graph."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, y):
        return self.m.loss(x, y)


# --------------------------------------------------------------------------- #
# training loop (shared with the llama baseline via train_wiki_llama.py)
# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--train-file", default="train2048.npy")
    ap.add_argument("--val-file", default="val2048.npy")
    ap.add_argument("--vocab", type=int, default=0,
                    help="0 = infer from bytes_mat/tokenizer path")
    ap.add_argument("--bytes-mat", default=None,
                    help="portable byte rows (V x n_bytes); V inferred from "
                         "its shape when --vocab 0")
    ap.add_argument("--n-bytes", type=int, default=8)
    ap.add_argument("--d-code", type=int, default=32)
    ap.add_argument("--codebook", default=None)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--n-attn", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--pos-mode", default="rope")
    ap.add_argument("--block-kind", default="gated")
    ap.add_argument("--allpos", action="store_true", default=True)
    ap.add_argument("--iso-lambda", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256, help="PER RANK")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta1", type=float, default=0.8)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--ckpt", default="ck_ddp.pt",
                    help="BEST checkpoint (overwritten only on improvement)")
    ap.add_argument("--log-every", type=int, default=50,
                    help="per-step train-loss line with ETA")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000,
                    help="periodic resume point at <ckpt>.last")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--ns-ortho", type=int, default=0,
                    help="Newton-Schulz re-orthogonalization of the depth-"
                         "chain block maps every N steps (0=off, try 50)")
    ap.add_argument("--world", type=int, default=2,
                    help="only used in the local (no-torchrun) fallback")
    return ap


def parse_args():
    return build_parser().parse_args()


_DEPTH_W = __import__("re").compile(r"blocks\.\d+\.(w1|w2|w3|weight)$")


def ns_polar_(W, iters=5):
    """In-place Newton-Schulz polar factor for a 2-D map of either
    orientation: works on the tall side (rows <= cols), so both wide (r, D)
    gated maps and tall (D, r) maps like w3 get orthonormal singular
    vectors. Keeps singular VALUES at 1, kills trained spectral drift; the
    gated skip (cos psi I + sin psi phi(.)) then stays isometric at ANY
    depth.
    """
    X = W.data.float()
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(iters):
        A = X.T @ X
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        X = 0.5 * X @ (3 * I - A)
    if tall:
        X = X.T
    W.data.copy_(X.to(W.dtype))


def ns_orthogonalize_(m, iters=5):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() == 2 and _DEPTH_W.search(name):
                ns_polar_(p, iters)


def run_ddp(m, args, rank, world, local, device, log_prefix="fip"):
    """The full train loop; m is the RAW (unwrapped) module."""
    main_rank = is_main(rank)
    W, B = args.window, args.batch

    # ---- data: mmap; one GPU-resident disjoint segment per rank
    tr = np.load(os.path.join(args.data_dir, args.train_file), mmap_mode="r")
    va = np.load(os.path.join(args.data_dir, args.val_file), mmap_mode="r")
    seg = len(tr) // world
    train = torch.from_numpy(np.array(tr[rank * seg:(rank + 1) * seg])) \
        .to(device).to(torch.int32)
    val = torch.from_numpy(np.array(va[:1_000_000])).to(device).to(torch.int32)
    log(rank, f"rank {rank}/{world}: train segment {seg/1e6:.1f}M tokens")

    def sample_all(data, count):
        pos = torch.randint(0, data.shape[0] - W - 1, (count,), device=device)
        off = torch.arange(W, device=device)
        idx = pos.unsqueeze(1) + off.unsqueeze(0)
        return data[idx].long(), data[idx + 1].long()

    vx, vy = sample_all(val, 2048)

    net = LossModule(m).to(device)
    dev_ids = [local] if device.type == "cuda" else None
    ddp = nn.parallel.DistributedDataParallel(net, device_ids=dev_ids) \
        if world > 1 else net
    lr_t = torch.tensor(args.lr, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=lr_t,
                           betas=(args.beta1, args.beta2),
                           fused=(device.type == "cuda"),
                           capturable=(device.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    step0, best, traj = 1, 0.0, []
    last_path = args.ckpt + ".last"
    for p in (args.ckpt, last_path):          # resume: prefer .last, fall back
        if os.path.exists(p):
            try:
                st = torch.load(p, map_location=device)
                m.load_state_dict(st["model"])
                opt.load_state_dict(st["opt"])
                step0, best, traj = st["step"] + 1, st["best"], st["traj"]
                log(rank, f"resumed at step {st['step']} from {p}")
                break
            except (RuntimeError, OSError, EOFError, KeyError):
                log(rank, f"ckpt {p} unreadable")
    jsonl = os.path.splitext(args.ckpt)[0] + ".jsonl"

    t0 = time.time()
    dt = 0.0
    for s in range(step0, args.steps + 1):
        ts = time.perf_counter()
        x, y = sample_all(train, B)
        with torch.amp.autocast("cuda", enabled=args.amp):
            loss = ddp(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        dt = (time.perf_counter() - ts) * 1000
        if args.ns_ortho and s % args.ns_ortho == 0:
            ns_orthogonalize_(m)
        # per-step progress: without this a run prints nothing until the
        # first eval and looks hung (worse under `!torchrun` buffering)
        if args.log_every and (s % args.log_every == 0 or s == step0):
            log(rank, f"  step {s:7d}  train {float(loss.detach()):.4f}"
                      f"  {dt:6.0f}ms"
                      f"  eta {(args.steps - s) * max(dt, 1) / 1000 / 60:5.1f}min")
        if s % args.eval_every == 0 or s == args.steps:
            # divergence guard: silent gradient non-reduction is the classic
            # DDP failure; make it loud
            if world > 1:
                probe = torch.tensor([float(next(m.parameters()).detach()
                                            .double().sum())],
                                     device=device, dtype=torch.float64)
                gat = [torch.zeros_like(probe) for _ in range(world)]
                dist.all_gather(gat, probe)
                spread = max(float(abs(g - gat[0])) for g in gat)
                if spread > 1e-6 * max(1.0, abs(float(gat[0]))):
                    log(rank, f"  *** DIVERGENCE at step {s}: checksum differs "
                               f"across ranks by {spread:.3e}. Gradients are "
                               f"NOT being reduced - results are invalid. ***")
            with torch.no_grad():
                pred = m.predict(vx)
                acc = (pred == vy).float().mean().item()
                acc_l = (pred[:, -1] == vy[:, -1]).float().mean().item()
                vloss = m.loss(vx, vy).item()
            torch.cuda.empty_cache()
            tag = ""
            if acc > best:
                best, tag = acc, "  *best*"
                save_ck(rank, args.ckpt, m, opt, s, world, args, best, traj)
            traj.append(f"{s//1000}k:{acc*100:.1f}")
            log(rank, f"  step {s:7d}  val {vloss:.3f}  acc(all) {acc*100:6.2f}%"
                      f"  acc(last) {acc_l*100:6.2f}%{tag}  {dt:.0f}ms")
            write_jsonl(rank, jsonl, {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": "eval",
                "step": s, "val_loss": vloss, "acc_all": acc, "acc_last": acc_l,
                "is_best": bool(tag), "train_loss": float(loss.detach()),
                "ms": dt, "lr": lr_t.item(),
            })
        if s % args.save_every == 0:
            save_ck(rank, last_path, m, opt, s, world, args, best, traj)
        if args.bench and s % 100 == 0:
            torch.cuda.synchronize()
            log(rank, f"bench {s}: {(time.time()-t0)/max(1, s-step0+1)*1000:.1f} ms/step/rank")
    save_ck(rank, last_path, m, opt, args.steps, world, args, best, traj)
    log(rank, f"{log_prefix} {args.steps} steps: BEST {best*100:.2f}% "
              f"[{' '.join(traj)}]")
    if world > 1:
        dist.destroy_process_group()
    return best


def main(args=None, rank=None, world=None):
    args = args or parse_args()
    rank, world, local, device = setup_distributed(rank, world)

    torch.manual_seed(0)
    bm = torch.load(args.bytes_mat, map_location="cpu")         if getattr(args, "bytes_mat", None) else None
    vocab = args.vocab or (int(bm.shape[0]) if bm is not None else 2048)
    m = av.AttLM(input_mode="byte", train_allpos=args.allpos,
                 pos_mode=args.pos_mode, block_kind=args.block_kind,
                 dim=args.dim, heads=args.heads, n_attn=args.n_attn,
                 depth_after=args.depth, code_dim=args.code_dim,
                 n_bytes=args.n_bytes, d_code=args.d_code,
                 vocab=vocab, window=args.window, bytes_mat=bm).to(device)
    if getattr(args, "iso_lambda", 0.0):
        m.iso_lambda = args.iso_lambda
    m.decoder.codes(cache_path=getattr(args, "codebook", None))
    params = sum(p.numel() for p in m.parameters())
    log(rank, f"model: dim {args.dim} depth {args.depth} n_attn {args.n_attn} "
              f"heads {args.heads} | {params/1e6:.2f}M params")
    run_ddp(m, args, rank, world, local, device)


def _spawn_worker(rank, args):
    """Local (no torchrun) multi-process entry: FileStore rendezvous."""
    os.environ["LOCAL_RANK"] = str(rank)
    main(args, rank=rank, world=args.world)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    _args = parse_args()
    if "RANK" in os.environ:                     # torchrun launch (Kaggle)
        main(_args)
    elif _args.world > 1:                        # local fallback: mp.spawn
        mp.spawn(_spawn_worker, args=(_args,), nprocs=_args.world, join=True)
    else:
        main(_args)
