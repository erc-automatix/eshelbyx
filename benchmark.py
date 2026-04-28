#!/usr/bin/env python3
"""
Side-by-side comparison of original vs. optimized implementations

This script:
1. Runs original implementation (float64)
2. Runs optimized implementation (float32)
3. Compares accuracy and performance
4. Generates comparison report

Usage:
    python3 compare_implementations.py --grid-size 64
    python3 compare_implementations.py --grid-size 100 --detailed
"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np
from time import time
import sys
from pathlib import Path

# Try to import GPU memory monitoring
try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True

    def get_gpu_memory():
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2
except:
    GPU_AVAILABLE = False

    def get_gpu_memory():
        return 0


def create_test_problem(grid_size):
    """Create a simple test problem for benchmarking"""
    from eshelbyx.grid import Grid
    from jaxmat.tensors import SymmetricTensor2, IsotropicTensor4

    # Simple homogeneous grid for testing
    markers = jnp.zeros((grid_size, grid_size, grid_size))
    grid = Grid(markers=markers, size=jnp.array([1.0, 1.0, 1.0]))

    # Reference material
    E0 = 1.0
    nu0 = 0.3
    lambda_0 = E0 * nu0 / (1 + nu0) / (1 - 2 * nu0)
    mu0 = E0 / 2 / (1 + nu0)
    C0 = IsotropicTensor4(lambda_0 + 2 / 3 * mu0, mu0)

    # Macroscopic strain
    E_macro = SymmetricTensor2(array=jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    E_macro = grid.broadcast_to_grid(E_macro)

    return grid, C0, E_macro


def run_baseline(grid, C0, E_macro, max_iter=100, verbose=False):
    """Run original float64 implementation"""
    from eshelbyx.green_operator import GreenOperator
    from eshelbyx.discretization import WillotDiscretization
    from eshelbyx.solver import AEMSolver

    # Configure for float64
    jax.config.update("jax_enable_x64", True)
    jax.clear_caches()

    print(f"Running BASELINE (float64)...")

    # Create operators
    Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=WillotDiscretization(grid))

    # Simple alpha function (identity for homogeneous case)
    def alpha(eps, scale=1.0):
        return eps

    # Warmup
    if verbose:
        print("  Warmup...")
    _ = AEMSolver(E_macro, alpha, Gamma_0, tol=1e-6, max_iter=5, verbose=False)
    jax.clear_caches()

    # Actual run
    if verbose:
        print("  Solving...")

    mem_before = get_gpu_memory()
    start = time()

    eps, sig, info = AEMSolver(
        E_macro, alpha, Gamma_0, tol=1e-6, max_iter=max_iter, verbose=False
    )
    jax.block_until_ready(eps)

    elapsed = time() - start
    mem_after = get_gpu_memory()

    return {
        "eps": np.array(eps),
        "sig": np.array(sig),
        "time": elapsed,
        "iterations": info.get("num_steps", info.get("iterations", 0)),
        "memory_mb": mem_after - mem_before,
        "peak_memory_mb": mem_after,
    }


def run_optimized(grid, C0, E_macro, max_iter=100, verbose=False):
    """Run optimized float32 implementation"""
    from eshelbyx.green_operator import GreenOperator
    from eshelbyx.discretization import WillotDiscretization
    from eshelbyx.solver import AEMSolver

    # Configure for float32
    jax.config.update("jax_enable_x64", False)
    jax.clear_caches()

    print(f"Running OPTIMIZED (float32)...")

    # Create operators
    Gamma_0 = GreenOperator(grid=grid, C0=C0, discretization=WillotDiscretization(grid))

    # Simple alpha function
    def alpha(eps, scale=1.0):
        return eps

    # Warmup
    if verbose:
        print("  Warmup...")
    _ = AEMSolver(E_macro, alpha, Gamma_0, tol=1e-6, max_iter=5, verbose=False)
    jax.clear_caches()

    # Actual run
    if verbose:
        print("  Solving...")

    mem_before = get_gpu_memory()
    start = time()

    eps, sig, info = AEMSolver(
        E_macro, alpha, Gamma_0, tol=1e-6, max_iter=max_iter, verbose=False
    )
    jax.block_until_ready(eps)

    elapsed = time() - start
    mem_after = get_gpu_memory()

    return {
        "eps": np.array(eps),
        "sig": np.array(sig),
        "time": elapsed,
        "iterations": info.get("num_steps", info.get("iterations", 0)),
        "memory_mb": mem_after - mem_before,
        "peak_memory_mb": mem_after,
    }


def compare_results(baseline, optimized, detailed=False):
    """Compare baseline and optimized results"""

    # Compute errors
    eps_error = np.linalg.norm(baseline["eps"] - optimized["eps"]) / np.linalg.norm(
        baseline["eps"]
    )
    sig_error = np.linalg.norm(baseline["sig"] - optimized["sig"]) / np.linalg.norm(
        baseline["sig"]
    )
    max_eps_error = np.max(np.abs(baseline["eps"] - optimized["eps"]))
    max_sig_error = np.max(np.abs(baseline["sig"] - optimized["sig"]))

    # Performance metrics
    speedup = baseline["time"] / optimized["time"]
    memory_reduction = (1 - optimized["memory_mb"] / baseline["memory_mb"]) * 100

    # Print comparison
    print(f"\n{'=' * 80}")
    print(f"COMPARISON RESULTS")
    print(f"{'=' * 80}\n")

    print(f"{'Metric':<30} {'Baseline':<20} {'Optimized':<20} {'Change':<15}")
    print(f"{'-' * 80}")

    print(
        f"{'Time (seconds)':<30} {baseline['time']:<20.3f} {optimized['time']:<20.3f} "
        f"{speedup:.2f}× faster"
    )

    print(
        f"{'Iterations':<30} {baseline['iterations']:<20} {optimized['iterations']:<20} "
        f"{baseline['iterations'] - optimized['iterations']:+d}"
    )

    if GPU_AVAILABLE:
        print(
            f"{'Memory used (MB)':<30} {baseline['memory_mb']:<20.1f} {optimized['memory_mb']:<20.1f} "
            f"{memory_reduction:.1f}% less"
        )
        print(
            f"{'Peak memory (MB)':<30} {baseline['peak_memory_mb']:<20.1f} "
            f"{optimized['peak_memory_mb']:<20.1f} "
            f"{(1 - optimized['peak_memory_mb'] / baseline['peak_memory_mb']) * 100:.1f}% less"
        )

    print(f"\n{'Accuracy Metrics':<30} {'Error':<20} {'Status':<20}")
    print(f"{'-' * 80}")

    eps_status = (
        "✓ Excellent"
        if eps_error < 1e-4
        else "✓ Good"
        if eps_error < 1e-3
        else "⚠ Check"
    )
    sig_status = (
        "✓ Excellent"
        if sig_error < 1e-4
        else "✓ Good"
        if sig_error < 1e-3
        else "⚠ Check"
    )

    print(f"{'Strain relative L2 error':<30} {eps_error:<20.6e} {eps_status:<20}")
    print(f"{'Stress relative L2 error':<30} {sig_error:<20.6e} {sig_status:<20}")
    print(f"{'Max strain error':<30} {max_eps_error:<20.6e}")
    print(f"{'Max stress error':<30} {max_sig_error:<20.6e}")

    if detailed:
        print(f"\n{'=' * 80}")
        print(f"DETAILED STATISTICS")
        print(f"{'=' * 80}\n")

        print(
            f"Baseline strain - mean: {np.mean(baseline['eps']):.6e}, std: {np.std(baseline['eps']):.6e}"
        )
        print(
            f"Optimized strain - mean: {np.mean(optimized['eps']):.6e}, std: {np.std(optimized['eps']):.6e}"
        )
        print(
            f"Baseline stress - mean: {np.mean(baseline['sig']):.6e}, std: {np.std(baseline['sig']):.6e}"
        )
        print(
            f"Optimized stress - mean: {np.mean(optimized['sig']):.6e}, std: {np.std(optimized['sig']):.6e}"
        )

    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}\n")

    if eps_error < 1e-3 and sig_error < 1e-3:
        print(f"✓ OPTIMIZATION SUCCESSFUL")
        print(f"  • Accuracy: Maintained (< 0.1% error)")
        print(f"  • Speed: {speedup:.2f}× faster")
        if GPU_AVAILABLE:
            print(f"  • Memory: {memory_reduction:.1f}% reduction")
        print(f"\n  You can now safely use float32 for this problem!")
    else:
        print(f"⚠ CAUTION ADVISED")
        print(f"  • Error is higher than expected")
        print(f"  • Consider adjusting solver tolerance or using selective float64")

    return {
        "eps_error": eps_error,
        "sig_error": sig_error,
        "speedup": speedup,
        "memory_reduction": memory_reduction,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs optimized implementations"
    )
    parser.add_argument(
        "--grid-size", type=int, default=64, help="Grid size for testing (default: 64)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum solver iterations (default: 100)",
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed statistics"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output during solve"
    )

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print(f"FFT Homogenization: Baseline vs. Optimized Comparison")
    print(f"{'=' * 80}")
    print(f"Grid size: {args.grid_size}³ ({args.grid_size**3:,} voxels)")
    print(f"Max iterations: {args.max_iter}")
    print(f"GPU available: {GPU_AVAILABLE}")
    print(f"{'=' * 80}\n")

    # Create test problem
    print("Creating test problem...")
    grid, C0, E_macro = create_test_problem(args.grid_size)
    print(f"Grid shape: {grid.shape}")
    print(f"Grid size: {grid.size}")

    # Run baseline
    print(f"\n{'=' * 80}")
    baseline = run_baseline(grid, C0, E_macro, args.max_iter, args.verbose)
    print(
        f"Baseline completed: {baseline['time']:.3f}s, {baseline['iterations']} iterations"
    )

    # Run optimized
    print(f"\n{'=' * 80}")
    optimized = run_optimized(grid, C0, E_macro, args.max_iter, args.verbose)
    print(
        f"Optimized completed: {optimized['time']:.3f}s, {optimized['iterations']} iterations"
    )

    # Compare
    metrics = compare_results(baseline, optimized, args.detailed)

    # Save results
    results_file = Path(f"comparison_results_{args.grid_size}.txt")
    with open(results_file, "w") as f:
        f.write(f"Grid size: {args.grid_size}³\n")
        f.write(f"Baseline time: {baseline['time']:.3f}s\n")
        f.write(f"Optimized time: {optimized['time']:.3f}s\n")
        f.write(f"Speedup: {metrics['speedup']:.2f}×\n")
        f.write(f"Memory reduction: {metrics['memory_reduction']:.1f}%\n")
        f.write(f"Strain error: {metrics['eps_error']:.6e}\n")
        f.write(f"Stress error: {metrics['sig_error']:.6e}\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
