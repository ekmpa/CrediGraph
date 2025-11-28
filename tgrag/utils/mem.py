import os

import psutil
import torch


def print_memory_usage(description: str = '') -> None:
    process = psutil.Process()
    print(f'{description}:')
    print(f'- RAM: {process.memory_info().rss / 1024**2:.2f} MB (System)')
    if torch.cuda.is_available():
        print(f'- GPU: {torch.cuda.memory_allocated() / 1024**2:.2f} MB (Allocated)')
        print(
            f'- GPU Cache: {torch.cuda.memory_reserved() / 1024**2:.2f} MB (Reserved)'
        )


def mem() -> None:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)  # MB
