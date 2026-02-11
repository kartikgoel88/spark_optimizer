"""
Pure functions for Spark 3.x optimization formulas.

Formulas are documented for auditability and tuning. All inputs come from
config or cluster specs; no hardcoded values except mathematical constants.
"""

import math
from typing import Optional

from spark_optimizer.config import ClusterConfig, WorkloadType
from spark_optimizer.constants import (
    CORES_RESERVED_PER_NODE,
    DRIVER_MEMORY_GB_DEFAULT,
    EXECUTOR_CORES_MAX,
    EXECUTOR_CORES_MIN,
    EXECUTOR_CORES_RECOMMENDED,
    EXECUTOR_MEMORY_GB_MAX,
    EXECUTOR_MEMORY_GB_MIN,
    MEMORY_RESERVED_GB_PER_NODE,
    PARTITIONS_MULTIPLIER_VS_CORES_MAX,
    PARTITIONS_MULTIPLIER_VS_CORES_MIN,
    SHUFFLE_PARTITIONS_MAX,
    SHUFFLE_PARTITIONS_MIN,
    WRITE_PARTITIONS_MAX,
    WRITE_PARTITIONS_MIN,
)


# ---------------------------------------------------------------------------
# Partition count for write (file size optimization)
# ---------------------------------------------------------------------------


def partitions_for_target_file_size(
    estimated_size_gb: float,
    target_file_size_mb: float,
    min_partitions: int = WRITE_PARTITIONS_MIN,
    max_partitions: int = WRITE_PARTITIONS_MAX,
) -> tuple[int, list[str]]:
    """
    Compute number of partitions so that each output file is ~target_file_size_mb.

    Formula:
        num_partitions = ceil(estimated_size_gb * 1024 / target_file_size_mb)

    Then clamped to [min_partitions, max_partitions] to avoid tiny or huge partition counts.

    Returns:
        (recommended_partitions, explanation_lines)
    """
    explanation: list[str] = []
    size_mb = estimated_size_gb * 1024.0
    raw = size_mb / target_file_size_mb
    n = max(1, math.ceil(raw))
    explanation.append(
        f"Partitions for file size: size_gb={estimated_size_gb}, target_mb={target_file_size_mb} "
        f"-> size_mb={size_mb:.0f}, raw_partitions={raw:.2f} -> ceil={n}"
    )
    n = max(min_partitions, min(max_partitions, n))
    if n != math.ceil(raw):
        explanation.append(f"Clamped to [{min_partitions}, {max_partitions}] -> {n}")
    return n, explanation


def cap_partitions_by_parallelism(
    num_partitions: int,
    total_executor_cores: int,
    multiplier_min: float = PARTITIONS_MULTIPLIER_VS_CORES_MIN,
    multiplier_max: float = PARTITIONS_MULTIPLIER_VS_CORES_MAX,
) -> tuple[int, list[str]]:
    """
    Cap partition count by cluster parallelism to avoid excessive tasks.

    Rule: partitions in [total_cores * multiplier_min, total_cores * multiplier_max]
    is reasonable. If num_partitions > total_cores * multiplier_max, cap to
    total_cores * multiplier_max for efficiency. If below multiplier_min * cores,
    we still return num_partitions (caller can enforce min if needed).

    Returns:
        (capped_partitions, explanation_lines)
    """
    explanation: list[str] = []
    cap = int(total_executor_cores * multiplier_max)
    floor = int(total_executor_cores * multiplier_min)
    if num_partitions > cap:
        explanation.append(
            f"Capping partitions by parallelism: {num_partitions} > {cap} "
            f"({total_executor_cores} cores * {multiplier_max}) -> {cap}"
        )
        return cap, explanation
    if num_partitions < floor and num_partitions > 0:
        explanation.append(
            f"Partitions {num_partitions} below recommended floor {floor} "
            f"({total_executor_cores} cores * {multiplier_min}); consider increasing for parallelism"
        )
    return num_partitions, explanation


# ---------------------------------------------------------------------------
# Shuffle partitions
# ---------------------------------------------------------------------------


def shuffle_partitions_for_workload(
    total_executor_cores: int,
    data_size_gb: float,
    workload_type: WorkloadType,
    shuffle_min: int = SHUFFLE_PARTITIONS_MIN,
    shuffle_max: int = SHUFFLE_PARTITIONS_MAX,
) -> tuple[int, list[str]]:
    """
    Recommend spark.sql.shuffle.partitions based on cores and workload.

    Formulas (Spark 3.x best practices):
    - ETL: 2x executor cores (moderate shuffle)
    - HEAVY_SHUFFLE: 3–4x executor cores, or scaled by data size
    - ML: 2x cores (balance with memory)
    - STREAMING: 1–2x cores (smaller batches)

    We also scale slightly by data size for very large datasets so shuffle
    doesn't become a bottleneck. Final value clamped to [shuffle_min, shuffle_max].

    Returns:
        (shuffle_partitions, explanation_lines)
    """
    explanation: list[str] = []
    multipliers = {
        WorkloadType.ETL: 2.0,
        WorkloadType.HEAVY_SHUFFLE: 4.0,
        WorkloadType.ML: 2.0,
        WorkloadType.STREAMING: 1.5,
    }
    mult = multipliers.get(workload_type, 2.0)
    n = int(total_executor_cores * mult)
    # For very large data, ensure enough partitions (e.g. 1 partition per ~2GB shuffle)
    if data_size_gb > 50 and workload_type == WorkloadType.HEAVY_SHUFFLE:
        size_based = min(shuffle_max, max(shuffle_min, int(data_size_gb * 4)))
        if size_based > n:
            n = size_based
            explanation.append(
                f"Heavy shuffle + large data ({data_size_gb} GB): raised to {n} for shuffle balance"
            )
    n = max(shuffle_min, min(shuffle_max, n))
    explanation.append(
        f"Shuffle partitions: {total_executor_cores} cores * {mult} ({workload_type.value}) -> {n} "
        f"(clamped to [{shuffle_min}, {shuffle_max}])"
    )
    return n, explanation


# ---------------------------------------------------------------------------
# Executor and cluster sizing (from node specs)
# ---------------------------------------------------------------------------


def executor_cores_per_node(
    node_cores: int,
    cores_reserved: int = CORES_RESERVED_PER_NODE,
    cores_min: int = EXECUTOR_CORES_MIN,
    cores_max: int = EXECUTOR_CORES_MAX,
    preferred: int = EXECUTOR_CORES_RECOMMENDED,
) -> tuple[int, list[str]]:
    """
    Choose executor cores per executor (same for all executors).

    Spark 3.x best practice: 4–5 cores per executor to reduce GC pressure.
    Formula: use preferred (5) if available, else fit (node_cores - reserved) / 1
    and clamp to [cores_min, cores_max].

    Returns:
        (executor_cores, explanation_lines)
    """
    explanation: list[str] = []
    available = node_cores - cores_reserved
    if available < 1:
        available = 1
    # Prefer recommended value but not more than available
    c = min(preferred, available, cores_max)
    c = max(cores_min, c)
    explanation.append(
        f"Executor cores: node_cores={node_cores}, reserved={cores_reserved}, "
        f"preferred={preferred} -> {c} (clamped [{cores_min}, {cores_max}])"
    )
    return c, explanation


def executors_per_node(
    node_cores: int,
    executor_cores: int,
    cores_reserved: int = CORES_RESERVED_PER_NODE,
) -> tuple[int, list[str]]:
    """
    Number of executors per worker node.

    Formula: floor((node_cores - cores_reserved) / executor_cores).
    Minimum 1.

    Returns:
        (executors_per_node, explanation_lines)
    """
    explanation: list[str] = []
    available = node_cores - cores_reserved
    if available < 1:
        available = 1
    n = max(1, available // executor_cores)
    explanation.append(
        f"Executors per node: (node_cores - reserved) / executor_cores = "
        f"({node_cores} - {cores_reserved}) / {executor_cores} -> {n}"
    )
    return n, explanation


def executor_memory_gb_per_node(
    node_memory_gb: float,
    executors_per_node: int,
    memory_reserved_gb: float = MEMORY_RESERVED_GB_PER_NODE,
    memory_min_gb: float = EXECUTOR_MEMORY_GB_MIN,
    memory_max_gb: float = EXECUTOR_MEMORY_GB_MAX,
) -> tuple[float, list[str]]:
    """
    Memory per executor (in GB) so that all executors fit on the node.

    Formula: (node_memory_gb - memory_reserved_gb) / executors_per_node
    Clamped to [memory_min_gb, memory_max_gb] to avoid tiny executors or huge heaps (GC).

    Returns:
        (executor_memory_gb, explanation_lines)
    """
    explanation: list[str] = []
    available = node_memory_gb - memory_reserved_gb
    if available < 1:
        available = 1.0
    mem = available / executors_per_node
    explanation.append(
        f"Executor memory: (node_memory - reserved) / executors_per_node = "
        f"({node_memory_gb} - {memory_reserved_gb}) / {executors_per_node} -> {mem:.2f} GB"
    )
    mem = max(memory_min_gb, min(memory_max_gb, mem))
    if mem != available / executors_per_node:
        explanation.append(f"Clamped to [{memory_min_gb}, {memory_max_gb}] -> {mem:.2f} GB")
    return round(mem, 2), explanation


def memory_overhead_mb(
    executor_memory_gb: float,
    overhead_ratio: float,
    overhead_min_mb: int,
) -> tuple[int, list[str]]:
    """
    Off-heap / overhead memory per executor (MB).

    Formula: max(executor_memory_gb * 1024 * overhead_ratio, overhead_min_mb)
    Spark uses this for off-heap, native memory, etc.

    Returns:
        (memory_overhead_mb, explanation_lines)
    """
    explanation: list[str] = []
    from_ratio = int(executor_memory_gb * 1024 * overhead_ratio)
    mb = max(overhead_min_mb, from_ratio)
    explanation.append(
        f"Memory overhead: max({executor_memory_gb} * 1024 * {overhead_ratio}, {overhead_min_mb}) -> {mb} MB"
    )
    return mb, explanation


def driver_memory_gb(
    data_size_gb: float,
    workload_type: WorkloadType,
    driver_min_gb: float = 1.0,
    driver_max_gb: float = 32.0,
    driver_default_gb: float = DRIVER_MEMORY_GB_DEFAULT,
) -> tuple[float, list[str]]:
    """
    Recommend driver memory. No exact formula; heuristic by workload and data size.

    - Small data / ETL: default (e.g. 4 GB)
    - Large data / heavy shuffle: increase (e.g. 8–16 GB) for broadcast/collect
    - ML: default–medium
    - STREAMING: moderate (driver manages state)

    Returns:
        (driver_memory_gb, explanation_lines)
    """
    explanation: list[str] = []
    gb = driver_default_gb
    if data_size_gb > 500 or workload_type == WorkloadType.HEAVY_SHUFFLE:
        gb = min(16.0, driver_default_gb * 2)
        explanation.append(f"Large data or heavy shuffle: driver memory -> {gb} GB")
    elif data_size_gb > 100:
        gb = min(8.0, driver_default_gb * 1.5)
        explanation.append(f"Medium-large data: driver memory -> {gb} GB")
    else:
        explanation.append(f"Driver memory: default {gb} GB")
    gb = max(driver_min_gb, min(driver_max_gb, gb))
    return round(gb, 2), explanation


def total_executor_cores(
    num_worker_nodes: int,
    executors_per_node: int,
    executor_cores: int,
) -> int:
    """Total executor cores across cluster = num_workers * executors_per_node * executor_cores."""
    return num_worker_nodes * executors_per_node * executor_cores
