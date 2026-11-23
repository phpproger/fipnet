"""
DDP training for AttLM (FIP resonance LM) - Kaggle T4 x 2 ready.

Launch (local single-host, works without torchrun/TCPStore):
  python experiments/ddp_train.py --world 2 --dim 320 --depth 10 ...

Launch via torchrun (Kaggle/cluster):
  torchrun --standalone --nproc_per_node=2 experiments/ddp_train.py \
      --dim 320 --depth 10 --n-attn 2 --steps 60000

Design notes:
  * Eager + DDP (CUDA graphs under DDP allreduce are fragile; on models this
    size the allreduce cost is negligible).
  * Data sharding: each rank samples windows from its disjoint segment of
    train2048.npy - no sampler sync, allpos targets stay local.
  * Checkpoints: rank 0 only, atomic (tmp + os.replace), saved WITHOUT the
    DDP "module." prefix; all ranks resume from the same file.
  * Memory hygiene: torch.cuda.empty_cache() after every eval.
  * Optional --amp: fp16 autocast + GradScaler (T4 fp16 tensor cores ~8x
    fp32 GEMM; phi-wave is bounded, so fp16 is safe-ish). Default off.
  * init: FileStore (portable, no libuv/TCPStore requirement) in the local
    spawn path; torchrun path inherits its own rendezvous.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiments.attention_v3 as av


_DEPTH_W = re.compile(r"blocks\.\d+\.(w1|w2|w3|weight)$")


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


class LossModule(nn.Module):
    """Wraps model.loss so DDP hooks see every parameter in the graph."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, y):
        return self.m.loss(x, y)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--train-file", default="train2048.npy",
                    help="e.g. train_tokens.npy for wikitext shards")
    ap.add_argument("--val-file", default="val2048.npy")
    ap.add_argument("--vocab", type=int, default=0,
                    help="0 = infer from bytes_mat/tokenizer path")
    ap.add_argument("--bytes-mat", default=None,
                    help="portable byte rows (V x n_bytes); V is inferred "
                         "from its shape when --vocab 0")
    ap.add_argument("--n-bytes", type=int, default=8)
    ap.add_argument("--d-code", type=int, default=32)
    ap.add_argument("--codebook", default=None)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--n-attn", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--code-dim", type=int, default=32)
    ap.add_argument("--pos-mode", default="rope")
    ap.add_argument("--block-kind", default="gated")
    ap.add_argument("--allpos", action="store_true", default=True)
    ap.add_argument("--iso-lambda", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256, help="per-rank batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt", default="ck_ddp.pt")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--ns-ortho", type=int, default=0,
                    help="Newton-Schulz re-orthogonalization of the depth-chain "
                         "block maps every N steps (0=off, try 50). Polar factor "
                         "of w1/w2/w3 keeps the isometric init anchor for deep "
                         "stacks (L>=14) against trained gain drift.")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="CPU smoke of DDP logic")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--world", type=int, default=2,
                    help="only used in local mp.spawn fallback")
    return ap.parse_args()


def setup(rank, world, store_path):
    """FileStore init: portable, no libuv/TCPStore needed."""
    if rank == 0 and os.path.exists(store_path):
        os.remove(store_path)
    backend = "nccl" if dist.is_nccl_available() else "gloo"   # Windows: gloo
    dist.init_process_group(backend,
                            init_method=f"file://{os.path.abspath(store_path)}",
                            rank=rank, world_size=world)
    local = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        local = min(local, torch.cuda.device_count() - 1)  # 1-GPU smoke: share
        torch.cuda.set_device(local)
    return rank, world, local


def main(rank, world, args=None, store_path=".ddp_store"):
    args = args or parse_args()
    rank, world, local = setup(rank, world, store_path)
    device = f"cuda:{local}" if torch.cuda.is_available() and not args.cpu else "cpu"
    is_main = rank == 0
    log = (lambda *a, **k: print(*a, flush=True, **k)) if is_main \
        else (lambda *a, **k: None)

    # ---- data: mmap; one GPU-resident disjoint segment per rank
    tr = np.load(os.path.join(args.data_dir, args.train_file), mmap_mode="r")
    va = np.load(os.path.join(args.data_dir, args.val_file), mmap_mode="r")
    n_tr = len(tr)
    seg = n_tr // world
    train = torch.from_numpy(np.array(tr[rank * seg:(rank + 1) * seg])) \
        .to(device).to(torch.int32)
    val = torch.from_numpy(np.array(va[:1_000_000])).to(device).to(torch.int32)
    W, B = args.window, args.batch
    log(f"rank {rank}/{world}: train segment {seg/1e6:.1f}M tokens")

    def sample_all(data, count):
        pos = torch.randint(0, data.shape[0] - W - 1, (count,), device=device)
        off = torch.arange(W, device=device)
        idx = pos.unsqueeze(1) + off.unsqueeze(0)
        return data[idx].long(), data[idx + 1].long()

    torch.manual_seed(0)
    vx, vy = sample_all(val, 2048)

    # ---- model (identical init on all ranks; DDP broadcasts anyway)
    torch.manual_seed(0)
    bm = torch.load(args.bytes_mat, map_location="cpu") if args.bytes_mat else None
    vocab = args.vocab or (int(bm.shape[0]) if bm is not None else 2048)
    m = av.AttLM(input_mode="byte", train_allpos=args.allpos,
                 pos_mode=args.pos_mode, block_kind=args.block_kind,
                 dim=args.dim, heads=args.heads, n_attn=args.n_attn,
                 depth_after=args.depth, code_dim=args.code_dim,
                 n_bytes=args.n_bytes, d_code=args.d_code,
                 vocab=vocab, window=W, bytes_mat=bm).to(device)
    if args.iso_lambda:
        m.iso_lambda = args.iso_lambda
    m.decoder.codes(cache_path=args.codebook)
    params = sum(p.numel() for p in m.parameters())
    log(f"model: dim {args.dim} depth {args.depth} n_attn {args.n_attn} "
        f"heads {args.heads} | {params/1e6:.2f}M params")

    net = LossModule(m).to(device)
    dev_ids = [local] if device.startswith("cuda") else None
    ddp = nn.parallel.DistributedDataParallel(net, device_ids=dev_ids) \
        if world > 1 else net
    opt = torch.optim.Adam(m.parameters(), lr=args.lr, betas=(0.8, 0.999),
                           fused=True, capturable=True)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # ---- resume
    step0, best, traj = 1, (0.0, 9e9), []
    if os.path.exists(args.ckpt):
        try:
            st = torch.load(args.ckpt, map_location=device)
            m.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            step0, best, traj = st["step"] + 1, st["best"], st["traj"]
            log(f"resumed at step {st['step']}")
        except (RuntimeError, OSError, EOFError):
            log("ckpt unreadable, fresh start")

    def eval_acc():
        with torch.no_grad():
            acc = (m.predict(vx) == vy).float().mean().item()
        torch.cuda.empty_cache()
        return acc

    def save(s, b, t):
        if not is_main:
            return
        tmp = args.ckpt + ".tmp"
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                    "step": s, "best": b, "traj": t}, tmp)
        os.replace(tmp, args.ckpt)

    t0 = time.time()
    for s in range(step0, args.steps + 1):
        x, y = sample_all(train, B)
        with torch.amp.autocast("cuda", enabled=args.amp):
            loss = ddp(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if args.ns_ortho and s % args.ns_ortho == 0:
            ns_orthogonalize_(m)
        if s % args.ckpt_every == 0:
            save(s, best, traj)
        if s % args.eval_every == 0:
            acc = eval_acc()
            best = (max(best[0], acc), best[1])
            traj.append(f"{s//1000}k:{acc*100:.1f}")
            log(f"step {s}: acc(all) {acc*100:.2f} [{time.time()-t0:.0f}s]")
        if args.bench and s % 100 == 0:
            torch.cuda.synchronize()
            log(f"bench {s}: {(time.time()-t0)/max(1, s-step0+1)*1000:.1f} ms/step/rank")
    acc = eval_acc()
    save(args.steps, best, traj)
    log(f"DDP[{args.dim}x{args.depth},na{args.n_attn}] {args.steps}: "
        f"BEST {best[0]*100:.2f}% params {params/1e6:.2f}M [{' '.join(traj)}]")
    dist.destroy_process_group()


def _worker(rank, args):
    os.environ["LOCAL_RANK"] = str(rank)
    main(rank, args.world, args)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    if "RANK" in os.environ:                     # torchrun launch (Kaggle)
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
    else:                                        # local fallback: mp.spawn
        args = parse_args()
        mp.spawn(_worker, args=(args,), nprocs=args.world, join=True)
