import torch
import time


def cuda_warmup(device='cuda', num_iters=1000000):
    if not torch.cuda.is_available():
        print("CUDA not available, skipping warm-up.")
        return

    device = torch.device(device)
    print(f"Warming up CUDA on {device}...")

    # Allocate some dummy tensors
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)

    # Simple computation to trigger GPU activity
    for i in range(num_iters):
        _ = torch.mm(x, y)
        torch.cuda.synchronize()  # Wait for GPU to finish work

    print("CUDA warm-up complete.")


if __name__ == "__main__":
    start = time.time()
    cuda_warmup()
    print(f"Warm-up took {time.time() - start:.2f} seconds.")
