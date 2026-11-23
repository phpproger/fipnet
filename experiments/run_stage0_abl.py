"""Stage-0-only ablation: plain CE(y_true) from scratch, the ONLY change is
the codebook: 'welch' (standard optimized, the 38.2% champion setup) vs
'gram' (Stage-0 aligned to the teacher head Gram, data-free). Isolates the
contribution of Stage 0 without any KD."""
import argparse, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["x"]
import experiments.attention_v3 as av
from experiments.mini_llama import MiniLlama
from experiments.tiny_stories import load_chunks
import experiments.distill as D

ap = argparse.ArgumentParser()
ap.add_argument("--codebook", default="welch", choices=["welch", "gram"])
args = ap.parse_args()
device = "cuda"
CKPT = rf"C:\Users\ignat\AppData\Local\Temp\ck_stage0_{args.codebook}.pt"
torch.manual_seed(0)
train, val = load_chunks(device, 60_000_000, 1_000_000)
W, B, STEPS = 64, 256, 30000
def sample_all(data, count):
    pos = torch.randint(0, data.shape[0]-W-1, (count,), device=device)
    off = torch.arange(W, device=device)
    idx = pos.unsqueeze(1)+off.unsqueeze(0)
    return data[idx].long(), data[idx+1].long()
torch.manual_seed(1)
vx, vy = sample_all(val, 2048)
torch.manual_seed(0)
m = av.AttLM(window=W, input_mode="byte", train_allpos=True, pos_mode="rope",
             block_kind="gated", dim=164, heads=1, n_attn=1, depth_after=8,
             code_dim=32).to(device)
m.iso_lambda = 1e-3
if args.codebook == "gram":
    T = MiniLlama(dim=160, layers=5, heads=4, allpos=True).to(device)
    st = torch.load(r"C:\Users\ignat\AppData\Local\Temp\ck_equal.pt",
                    map_location=device)
    T.load_state_dict(st["model"])
    D.align_codebook(m, D.teacher_logits_w(T).detach(), steps=1500,
                     device=device)
    del T
    torch.cuda.empty_cache()
else:
    m.decoder.codes(cache_path="data/codebook_2048_fourier32.pt")
params = sum(p.numel() for p in m.parameters())
opt = torch.optim.Adam(m.parameters(), lr=3e-4, betas=(0.8, 0.999),
                       fused=True, capturable=True)
from fipnet.graph_step import GraphedStep
gs = GraphedStep(m, opt, vx[:B], vy[:B])
start, best, traj = 1, (0.0, 9e9), []
if os.path.exists(CKPT):
    try:
        st = torch.load(CKPT, map_location=device)
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start, best, traj = st["step"]+1, st["best"], st["traj"]
        print(f"resumed at {st['step']}", flush=True)
    except (RuntimeError, OSError, EOFError):
        print("ckpt incompatible, fresh", flush=True); os.remove(CKPT)
t0 = time.time()
for s in range(start, STEPS + 1):
    x, y = sample_all(train, B)
    gs.step(x, y)
    if s % 1000 == 0:
        tmp = CKPT + ".tmp"
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                    "step": s, "best": best, "traj": traj}, tmp)
        os.replace(tmp, CKPT)
    if s % 5000 == 0:
        with torch.no_grad():
            acc = (m.predict(vx) == vy).float().mean().item()
            vloss = m.loss(vx, vy).item()
        torch.cuda.empty_cache()
        best = (max(best[0], acc), min(best[1], vloss))
        traj.append(f"{s//1000}k:{acc*100:.1f}")
        print(f"step {s}: acc(all) {acc*100:.2f} loss {vloss:.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
print(f"STAGE0-ABL[{args.codebook}] 30k: BEST {best[0]*100:.2f}% loss {best[1]:.3f} "
      f"params {params/1e6:.2f}M [{' '.join(traj)}]")
