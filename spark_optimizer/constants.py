"""
Safety limits and default bounds for Spark 3.x tuning.

These are upper/lower bounds to prevent over-allocation and GC-heavy configs.
All values can be overridden via config; these are production-oriented defaults.
"""

from typing import Any

# --- Executor & cluster ---
EXECUTOR_CORES_MIN = 1
EXECUTOR_CORES_MAX = 8  # >5 often increases GC; 4–5 is sweet spot for many workloads
EXECUTOR_CORES_RECOMMENDED = 5  # Spark 3.x best practice to limit GC
EXECUTOR_MEMORY_GB_MAX = 64  # Very large heaps => long GC pauses
EXECUTOR_MEMORY_GB_MIN = 2
MEMORY_OVERHEAD_RATIO_MAX = 0.40
MEMORY_OVERHEAD_RATIO_MIN = 0.05
MEMORY_OVERHEAD_MIN_MB = 384  # Spark default
DRIVER_MEMORY_GB_MIN = 1
DRIVER_MEMORY_GB_MAX = 32
DRIVER_MEMORY_GB_DEFAULT = 4

# --- Partitions ---
SHUFFLE_PARTITIONS_MIN = 1
SHUFFLE_PARTITIONS_MAX = 10000
SHUFFLE_PARTITIONS_DEFAULT = 200  # Spark default
WRITE_PARTITIONS_MIN = 1
WRITE_PARTITIONS_MAX = 100_000  # Avoid millions of tiny tasks
PARTITIONS_MULTIPLIER_VS_CORES_MIN = 1.0  # At least 1 partition per core
PARTITIONS_MULTIPLIER_VS_CORES_MAX = 4.0  # 2–4x cores common for parallelism

# --- Node reservation (for YARN/k8s) ---
CORES_RESERVED_PER_NODE = 1  # OS / AM / daemons
MEMORY_RESERVED_GB_PER_NODE = 1  # OS / system

# --- File size ---
TARGET_FILE_SIZE_MB_DEFAULT = 128
TARGET_FILE_SIZE_MB_MIN = 64
TARGET_FILE_SIZE_MB_MAX = 512


def get_default_limits() -> dict[str, Any]:
    """Return a dict of default limits for use in config or tests."""
    return {
        "executor_cores_min": EXECUTOR_CORES_MIN,
        "executor_cores_max": EXECUTOR_CORES_MAX,
        "executor_cores_recommended": EXECUTOR_CORES_RECOMMENDED,
        "executor_memory_gb_max": EXECUTOR_MEMORY_GB_MAX,
        "executor_memory_gb_min": EXECUTOR_MEMORY_GB_MIN,
        "memory_overhead_ratio_max": MEMORY_OVERHEAD_RATIO_MAX,
        "memory_overhead_ratio_min": MEMORY_OVERHEAD_RATIO_MIN,
        "memory_overhead_min_mb": MEMORY_OVERHEAD_MIN_MB,
        "driver_memory_gb_min": DRIVER_MEMORY_GB_MIN,
        "driver_memory_gb_max": DRIVER_MEMORY_GB_MAX,
        "driver_memory_gb_default": DRIVER_MEMORY_GB_DEFAULT,
        "shuffle_partitions_min": SHUFFLE_PARTITIONS_MIN,
        "shuffle_partitions_max": SHUFFLE_PARTITIONS_MAX,
        "shuffle_partitions_default": SHUFFLE_PARTITIONS_DEFAULT,
        "write_partitions_min": WRITE_PARTITIONS_MIN,
        "write_partitions_max": WRITE_PARTITIONS_MAX,
        "partitions_multiplier_vs_cores_min": PARTITIONS_MULTIPLIER_VS_CORES_MIN,
        "partitions_multiplier_vs_cores_max": PARTITIONS_MULTIPLIER_VS_CORES_MAX,
        "cores_reserved_per_node": CORES_RESERVED_PER_NODE,
        "memory_reserved_gb_per_node": MEMORY_RESERVED_GB_PER_NODE,
        "target_file_size_mb_default": TARGET_FILE_SIZE_MB_DEFAULT,
        "target_file_size_mb_min": TARGET_FILE_SIZE_MB_MIN,
        "target_file_size_mb_max": TARGET_FILE_SIZE_MB_MAX,
    }
