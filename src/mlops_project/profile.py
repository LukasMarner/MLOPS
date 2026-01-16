"""Profiling utilities for performance optimization."""

import cProfile
import pstats
from pathlib import Path
from typing import Callable, Optional


def profile_function(
    func: Callable,
    output_file: Optional[Path] = None,
    sort_by: str = "cumulative"
) -> pstats.Stats:
    """Profile a function and return statistics.

    Args:
        func: Function to profile
        output_file: Optional file to save profile results
        sort_by: Sort key for stats (cumulative, time, calls, etc.)

    Returns:
        Stats object with profiling results

    Example:
        >>> def my_function():
        ...     # some code
        ...     pass
        >>> stats = profile_function(my_function, output_file="profile.txt")
        >>> stats.print_stats(10)  # Print top 10 functions
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        result = func()
    finally:
        profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats(sort_by)
    
    if output_file:
        with open(output_file, 'w') as f:
            stats.print_stats(file=f)
        print(f"Profile saved to {output_file}")
    
    return stats


def profile_training_step(model, data_loader, device, num_batches: int = 10):
    """Profile a training step for performance analysis.

    Args:
        model: PyTorch model
        data_loader: DataLoader to iterate over
        num_batches: Number of batches to profile

    Returns:
        Stats object with profiling results
    """
    import torch
    import torch.nn.functional as F
    
    model.train()
    
    def training_iteration():
        for i, (images, labels) in enumerate(data_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Clear gradients (simulate optimizer step)
            model.zero_grad()
    
    return profile_function(training_iteration)


if __name__ == "__main__":
    # Example usage
    print("Profiling utilities available. Import and use profile_function() or profile_training_step()")

