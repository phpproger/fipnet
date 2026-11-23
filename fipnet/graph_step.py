"""CUDA-graph capture of a full training step (zero_grad + fwd + bwd + Adam).

Static shapes required; the batch is copied into static buffers before each
replay. Use Adam(fused=True, capturable=True). Removes kernel-launch
overhead entirely — essential for small models made of many tiny kernels.
"""

import torch


class GraphedStep:
    def __init__(self, model, opt, example_x, example_y):
        self.model, self.opt = model, opt
        self.sx = example_x.clone()
        self.sy = example_y.clone()
        for p in model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):                      # warmup in the graph pool
                self.opt.zero_grad(set_to_none=False)
                self.static_loss = self.model.loss(self.sx, self.sy)
                self.static_loss.backward()
                self.opt.step()
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.opt.zero_grad(set_to_none=False)
            self.static_loss = self.model.loss(self.sx, self.sy)
            self.static_loss.backward()
            self.opt.step()

    def step(self, x, y):
        """Copy the batch into static buffers and replay the captured step."""
        self.sx.copy_(x, non_blocking=True)
        self.sy.copy_(y, non_blocking=True)
        self.g.replay()
        return self.static_loss.detach()
