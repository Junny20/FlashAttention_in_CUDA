
import torch
import torch.nn.functional as F

N, d = 1024, 64

Q = torch.randn(N, d, device="cuda")
K = torch.randn(N, d, device="cuda")
V = torch.randn(N, d, device="cuda")

def bench_torch_cuda_events(iters=100):
    q = Q.unsqueeze(0).unsqueeze(0)
    k = K.unsqueeze(0).unsqueeze(0)
    v = V.unsqueeze(0).unsqueeze(0)

    for _ in range(20):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = F.scaled_dot_product_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return total_ms / iters

avg = bench_torch_cuda_events(100000)
print(f"PyTorch sdpa: {avg:.4f} ms")
