"""DDP training for the MiniLlama baseline (Kaggle T4 x 2) - same protocol as
ddp_train.py but the classic llama stack (RMSNorm + RoPE + SwiGLU + tied head).

Iso-fair 10M pair for wikitext-103 (V=8192):
  FIP:   experiments/ddp_train.py --dim 640 --depth 14 --n-attn 2 --heads 4
         = 9.87M params (0 vocab params; ~0.26M codebook dictionary)
  llama: experiments/train_wiki_llama.py --dim 224 --layers 13 --heads 8
         = 9.67M params (1.84M vocab params, tied head)

Launch:
  torchrun --standalone --nproc_per_node=2 experiments/train_wiki_llama.py \
      --dim 224 --layers 13 --heads 8 --vocab 8192 \
      --train-file train_tokens.npy --val-file val_tokens.npy \
      --steps 50000 --batch 512 --ckpt /kaggle/working/ck_wiki_llama10m.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.mini_llama import MiniLlama
from experiments.ddp_train import LossModule, setup


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--train-file", default="train2048.npy")
    ap.add_argument("--val-file", default="val2048.npy")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--dim", type=int, default=224)
    ap.add_argument("--layers", type=int, default=13)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tie", action="store_true", default=True)
    ap.add_argument("--no-tie", dest="tie", action="store_false")
    ap.add_argument("--allpos", action="store_true", default=True)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=256, help="per-rank batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt", default="ck_llama_ddp.pt")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="CPU smoke of DDP logic")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--world", type=int, default=2,
                    help="only used in local mp.spawn fallback")
    return ap.parse_args()


def main(rank, world, args=None, store_path=".ddp_store_llama"):
    args = args or parse_args()
    rank, world, local = setup(rank, world, store_path)
    device = f"cuda:{local}" if torch.cuda.is_available() and not args.cpu else "cpu"
    is_main = rank == 0
    log = (lambda *a, **k: print(*a, flush=True, **k)) if is_main \
        else (lambda *a, **k: None)

    tr = np.load(os.path.join(args.data_dir, args.train_file), mmap_mode="r")
    va = np.load(os.path.join(args.data_dir, args.val_file), mmap_mode="r")
    seg = len(tr) // world
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
    m = MiniLlama(dim=args.dim, layers=args.layers, heads=args.heads,
                  window=W, tie=args.tie, allpos=args.allpos,
                  vocab=args.vocab).to(device)
    params = sum(p.numel() for p in m.parameters())
    n_vocab = m.embed.weight.numel() + (m.head.weight.numel() if m.head else 0)
    log(f"llama: dim {args.dim} layers {args.layers} heads {args.heads} "
        f"{'tied' if args.tie else 'untied'} | {params/1e6:.2f}M params "
        f"(vocab {n_vocab/1e6:.2f}M)")

    net = LossModule(m).to(device)
    dev_ids = [local] if device.startswith("cuda") else None
    ddp = nn.parallel.DistributedDataParallel(net, device_ids=dev_ids) \
        if world > 1 else net
    opt = torch.optim.Adam(m.parameters(), lr=args.lr, betas=(0.8, 0.999),
                           fused=True, capturable=True)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

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

    vx, vy = sample_all(val, 2048)
    t0 = time.time()
    for s in range(step0, args.steps + 1):
        x, y = sample_all(train, B)
        with torch.amp.autocast("cuda", enabled=args.amp):
            loss = ddp(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
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
    log(f"llama[{args.dim}x{args.layers}] {args.steps}: BEST {best[0]*100:.2f}% "
        f"params {params/1e6:.2f}M [{' '.join(traj)}]")
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
