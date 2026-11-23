"""TinyStories BPE-2048 window-LM experiment: FIP-Net vs embedding MLP.

Data: pre-encoded BPE-2048 corpus (tinystories_full_bpe2048), 499.7M train
tokens, 5.0M val tokens as uint16 memmaps. Task: predict the next token
from a fixed window of W=64 previous tokens (Bengio-style window LM).

FIP model: structured dyadic harmonic codes (omega_j = 2^j / V with
quadrature T/4 pairs) lifted by one shared Linear(2*n_oct -> D), gathered
from a per-step V x D LUT — no embedding table, no V x D output matrix.
Position info via flatten projection; isometric blocks; low-rank phase
bottleneck head (K=16 unit-norm quad codes), resonance decoding.
Training: fused capturable Adam inside a CUDA graph (static shapes).

Usage: python experiments/tiny_stories.py [--steps 30000] [--window 64]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from fipnet.model import FIPWindowLM, MlpWindowLM
from fipnet.graph_step import GraphedStep

DATA_DIR = Path(r"C:\Users\ignat\PycharmProjects\future\tinystories_full_bpe2048")


def load_chunks(device, train_tokens=60_000_000, val_tokens=2_000_000):
    tr = np.load(DATA_DIR / "train2048.npy", mmap_mode="r")
    va = np.load(DATA_DIR / "val2048.npy", mmap_mode="r")
    # uint16 -> int32: CUDA lacks uint16 indexing; int32 keeps memory low
    train = torch.from_numpy(np.array(tr[:train_tokens])).to(device).to(torch.int32)
    val = torch.from_numpy(np.array(va[:val_tokens])).to(device).to(torch.int32)
    return train, val


def run(name, model, train, val, window, steps, batch, device, lr, use_graph):
    torch.manual_seed(0)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, fused=True, capturable=use_graph)

    def sample(data, count):
        pos = torch.randint(0, data.shape[0] - window - 1, (count,), device=device)
        off = torch.arange(window, device=device)
        x = data[pos.unsqueeze(1) + off.unsqueeze(0)].long()
        y = data[pos + window].long()
        return x, y

    torch.manual_seed(1)
    vx, vy = sample(val, 4096)
    torch.manual_seed(0)

    stepper = GraphedStep(model, opt, vx[:batch], vy[:batch]) if use_graph else None

    t0 = time.time()
    best_acc, n_steps = 0.0, 0
    for step in range(1, steps + 1):
        x, y = sample(train, batch)
        if use_graph:
            loss = stepper.step(x, y)
        else:
            opt.zero_grad(set_to_none=True)
            loss = model.loss(x, y)
            loss.backward()
            opt.step()
        n_steps += 1
        if step % 2000 == 0 or step == 1:
            with torch.no_grad():
                vacc = (model.predict(vx) == vy).float().mean().item()
                vloss = model.loss(vx, vy).item()
            best_acc = max(best_acc, vacc)
            print(f"  [{name}] step {step}: val acc {vacc*100:.2f}% "
                  f"val loss {vloss:.3f}  ({(time.time()-t0)/n_steps*1000:.2f} ms/step)",
                  flush=True)
    return dict(name=name, acc=best_acc,
                params=sum(p.numel() for p in model.parameters()),
                seconds=time.time() - t0, ms_per_step=(time.time() - t0) / n_steps * 1000)


@torch.no_grad()
def generate(model, tokenizer, train, window, device, n_tokens=120, temperature=0.8):
    import torch.nn.functional as Fm
    model.eval()
    seed = "Once upon a time, there was a"
    ids = tokenizer.encode(seed).ids[-window:]
    ctx = torch.full((1, window), 2, dtype=torch.long, device=device)
    for i, t in enumerate(ids):
        ctx[0, window - len(ids) + i] = t
    out = seed
    for _ in range(n_tokens):
        logits = model.decoder.head(model.features(ctx))
        probs = Fm.softmax(logits[0] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        out += tokenizer.decode([nxt])
        ctx = torch.roll(ctx, -1, dims=1)
        ctx[0, -1] = nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--generate", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_graph = (device == "cuda") and not args.no_graph
    V = 2048
    print(f"device: {device}, vocab {V}, window {args.window}, cuda-graph: {use_graph}")
    train, val = load_chunks(device)

    results = []
    results.append(run("fip-windowlm",
                       FIPWindowLM(V, window=args.window, dim=args.dim, depth=6,
                                   code_dim=32, n_oct=11, decoder_mode="fourier"),
                       train, val, args.window, args.steps, args.batch, device,
                       args.lr, use_graph))
    results.append(run("mlp-windowlm-baseline",
                       MlpWindowLM(V, window=args.window, dim=args.dim, depth=6),
                       train, val, args.window, args.steps, args.batch, device,
                       args.lr, use_graph))

    print("\n=== TinyStories BPE-2048 summary (best val next-token accuracy) ===")
    print(f"{'model':24s} {'acc %':>7s} {'params':>9s} {'ms/step':>8s} {'sec':>7s}")
    for r in results:
        print(f"{r['name']:24s} {r['acc']*100:7.2f} {r['params']:9d} "
              f"{r['ms_per_step']:8.2f} {r['seconds']:7.1f}")

    if args.generate:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(r"C:\Users\ignat\PycharmProjects\glm\data\tinystories_tokenizer.json")
        model = FIPWindowLM(V, window=args.window, dim=args.dim, depth=6,
                            code_dim=32, n_oct=11, decoder_mode="fourier").to(device)
        torch.manual_seed(0)
        print("\n=== FIP sample generation ===")
        print(generate(model, tk, train, args.window, device))


if __name__ == "__main__":
    main()
