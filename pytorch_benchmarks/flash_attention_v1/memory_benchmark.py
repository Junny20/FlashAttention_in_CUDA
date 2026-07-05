
import torch
import torch.nn.functional as F

def naive_attention(Q, K, V):
    d = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / (d ** 0.5)   # materializes N×N — the big allocation
    P = torch.softmax(scores, dim=-1)                  # another N×N
    return P @ V

for N in [512, 1024, 2048]:
    d = 64
    Q = torch.randn(N, d, device='cuda')
    K = torch.randn(N, d, device='cuda')
    V = torch.randn(N, d, device='cuda')

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = naive_attention(Q, K, V)
    torch.cuda.synchronize()
    naive_peak = torch.cuda.max_memory_allocated() / 1e6   # MB
    print(f"N={N}: naive peak = {naive_peak:.2f} MB")
