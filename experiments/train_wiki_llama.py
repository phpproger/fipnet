"""DDP training for the MiniLlama baseline - the same proven loop as
ddp_train.py (env rendezvous, per-step progress, JSONL log, best+last
ckpts, divergence guard). Classic llama stack: RMSNorm + RoPE + SwiGLU
(+ tied head).

Iso-fair 10M pair for wikitext-103 (V=8192):
  FIP:   experiments/ddp_train.py --dim 640 --depth 14 --n-attn 2 --heads 4
         = ~10.1M params (0 vocab params)
  llama: experiments/train_wiki_llama.py --dim 224 --layers 13 --heads 8
         = 9.67M params (1.84M vocab params, tied head)

Launch:
  torchrun --standalone --nproc_per_node=2 experiments/train_wiki_llama.py \
      --dim 224 --layers 13 --heads 8 --vocab 8192 --tie \
      --train-file train_tokens.npy --val-file val_tokens.npy \
      --steps 50000 --batch 512 --ckpt /kaggle/working/ck_wiki_llama10m.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.mini_llama import MiniLlama
from experiments.ddp_train import (LossModule, build_parser, run_ddp,
                                   setup_distributed, log)


def parse_args():
    ap = argparse.ArgumentParser(
        parents=[build_parser()], add_help=False,
        conflict_handler="resolve")     # child overrides shared defaults
    ap.set_defaults(vocab=8192, beta1=0.9, beta2=0.95, iso_lambda=0.0,
                    log_every=50, eval_every=1000, save_every=2000)
    ap.add_argument("--layers", type=int, default=13)
    ap.add_argument("--tie", action="store_true", default=True)
    ap.add_argument("--no-tie", dest="tie", action="store_false")
    return ap.parse_args()


def main(args=None, rank=None, world=None):
    args = args or parse_args()
    rank, world, local, device = setup_distributed(rank, world)

    torch.manual_seed(0)
    m = MiniLlama(dim=args.dim, layers=args.layers, heads=args.heads,
                  window=args.window, tie=args.tie, allpos=args.allpos,
                  vocab=args.vocab).to(device)
    params = sum(p.numel() for p in m.parameters())
    n_vocab = m.embed.weight.numel() + (m.head.weight.numel() if m.head else 0)
    log(rank, f"llama: dim {args.dim} layers {args.layers} heads {args.heads} "
              f"{'tied' if args.tie else 'untied'} | {params/1e6:.2f}M params "
              f"(vocab {n_vocab/1e6:.2f}M)")
    run_ddp(m, args, rank, world, local, device, log_prefix="llama")


def _spawn_worker(rank, args):
    os.environ["LOCAL_RANK"] = str(rank)
    main(args, rank=rank, world=args.world)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    _args = parse_args()
    if "RANK" in os.environ:
        main(_args)
    elif _args.world > 1:
        mp.spawn(_spawn_worker, args=(_args,), nprocs=_args.world, join=True)
    else:
        main(_args)
