#!/usr/bin/env python3
"""
Test script to verify GPU usage
"""
import torch
import time
import subprocess

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create a simple tensor operation on GPU
    print("\nTesting GPU computation...")
    device = torch.device("cuda")
    
    # Create large tensors to ensure GPU usage is visible
    size = 10000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    print(f"Tensor a device: {a.device}")
    print(f"Tensor b device: {b.device}")
    
    # Check GPU memory before computation
    print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Run nvidia-smi before computation
    try:
        print("\nnvidia-smi before computation:")
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except:
        pass
    
    # Perform matrix multiplication
    print("\nPerforming matrix multiplication...")
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # Wait for computation to complete
    end = time.time()
    
    print(f"Computation time: {end - start:.2f} seconds")
    print(f"Result device: {c.device}")
    print(f"Result shape: {c.shape}")
    
    # Check GPU memory after computation
    print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Run nvidia-smi after computation
    try:
        print("\nnvidia-smi after computation:")
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except:
        pass
    
else:
    print("CUDA is not available!")