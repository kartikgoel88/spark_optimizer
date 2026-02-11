"""
Cluster configuration calculator for Spark 3.x.

Computes optimal executors, cores, memory, overhead, driver memory, and
shuffle partitions from node specs, workload type, and data size.
Includes safety guards and human-readable explanations.
"""

from typing import Any, Optional

from spark_optimizer.config import (
    ClusterConfig,
    SparkSessionConfig,
    WorkloadType,
)
from spark_optimizer.constants import (
    EXECUTOR_CORES_MAX,
    EXECUTOR_CORES_MIN,
    EXECUTOR_CORES_RECOMMENDED,
    EXECUTOR_MEMORY_GB_MAX,
    EXECUTOR_MEMORY_GB_MIN,
    WRITE_PARTITIONS_MAX,
    WRITE_PARTITIONS_MIN,
)
from spark_optimizer.formulas import (
    cap_partitions_by_parallelism,
    driver_memory_gb,
    executor_cores_per_node,
    executor_memory_gb_per_node,
    executors_per_node,
    memory_overhead_mb,
    partitions_for_target_file_size,
    shuffle_partitions_for_workload,
    total_executor_cores,
)


class ClusterRecommendation:
    """Result of cluster calculation: config dict + explanation + safety notes."""

    def __init__(
        self,
        config: SparkSessionConfig,
        total_executor_cores: int,
        num_executors: int,
        safety_notes: list[str],
    ):
        self.config = config
        self.total_executor_cores = total_executor_cores
        self.num_executors = num_executors
        self.safety_notes = safety_notes

    def to_spark_config_dict(self) -> dict[str, Any]:
        """Flatten to a single dict suitable for SparkConf or spark.conf."""
        out: dict[str, Any] = dict(self.config.config_dict)
        if self.config.shuffle_partitions is not None:
            out["spark.sql.shuffle.partitions"] = str(self.config.shuffle_partitions)
        return out


class ClusterCalculator:
    """
    Computes recommended Spark configuration from cluster and workload inputs.

    Inputs:
        - node_memory_gb, node_cores, num_worker_nodes (or full ClusterConfig)
        - workload_type: ETL | heavy_shuffle | ML | streaming
        - data_size_gb: estimated data size in GB

    Outputs:
        - Recommended Spark config dictionary
        - Explanation of each calculation
        - Safety guards to prevent over-allocation
    """

    def __init__(self, limits: Optional[dict[str, Any]] = None):
        """
        Args:
            limits: Optional overrides for constants (e.g. executor_cores_max, driver_memory_gb_max).
                    If None, uses defaults from constants module.
        """
        self._limits = limits or {}

    def _get(self, key: str, default: Any) -> Any:
        return self._limits.get(key, default)

    def recommend(
        self,
        node_memory_gb: float,
        node_cores: int,
        num_worker_nodes: int,
        workload_type: WorkloadType,
        data_size_gb: float,
        *,
        target_file_size_mb: Optional[float] = None,
        executor_cores_override: Optional[int] = None,
        executor_memory_gb_override: Optional[float] = None,
        driver_memory_gb_override: Optional[float] = None,
    ) -> ClusterRecommendation:
        """
        Compute full Spark configuration recommendation.

        Returns:
            ClusterRecommendation with config_dict, explanation, and safety_notes.
        """
        from spark_optimizer.constants import (
            CORES_RESERVED_PER_NODE,
            DRIVER_MEMORY_GB_DEFAULT,
            MEMORY_RESERVED_GB_PER_NODE,
        )

        explanation: list[str] = []
        safety_notes: list[str] = []

        # 1) Executor cores
        preferred = self._get("executor_cores_recommended", EXECUTOR_CORES_RECOMMENDED)
        cores_min = self._get("executor_cores_min", EXECUTOR_CORES_MIN)
        cores_max = self._get("executor_cores_max", EXECUTOR_CORES_MAX)
        if executor_cores_override is not None:
            executor_cores = max(cores_min, min(cores_max, executor_cores_override))
            explanation.append(f"Executor cores (override): {executor_cores}")
        else:
            executor_cores, lines = executor_cores_per_node(
                node_cores,
                cores_reserved=self._get("cores_reserved_per_node", CORES_RESERVED_PER_NODE),
                cores_min=cores_min,
                cores_max=cores_max,
                preferred=preferred,
            )
            explanation.extend(lines)

        # 2) Executors per node
        execs_per_node, lines = executors_per_node(
            node_cores,
            executor_cores,
            cores_reserved=self._get("cores_reserved_per_node", CORES_RESERVED_PER_NODE),
        )
        explanation.extend(lines)
        num_executors = num_worker_nodes * execs_per_node
        if num_executors <= 0:
            num_executors = 1
            safety_notes.append("num_executors was <= 0; set to 1")
        explanation.append(f"Total executors: {num_worker_nodes} nodes * {execs_per_node} = {num_executors}")

        # 3) Executor memory
        mem_reserved = self._get("memory_reserved_gb_per_node", MEMORY_RESERVED_GB_PER_NODE)
        mem_min = self._get("executor_memory_gb_min", EXECUTOR_MEMORY_GB_MIN)
        mem_max = self._get("executor_memory_gb_max", EXECUTOR_MEMORY_GB_MAX)
        if executor_memory_gb_override is not None:
            executor_memory_gb = max(mem_min, min(mem_max, executor_memory_gb_override))
            explanation.append(f"Executor memory (override): {executor_memory_gb} GB")
        else:
            executor_memory_gb, lines = executor_memory_gb_per_node(
                node_memory_gb,
                execs_per_node,
                memory_reserved_gb=mem_reserved,
                memory_min_gb=mem_min,
                memory_max_gb=mem_max,
            )
            explanation.extend(lines)
        if executor_memory_gb > 32:
            safety_notes.append(
                "Executor memory > 32 GB may increase GC pauses; consider multiple smaller executors"
            )

        # 4) Memory overhead
        overhead_ratio = 0.10
        overhead_min_mb = 384
        overhead_mb, lines = memory_overhead_mb(
            executor_memory_gb,
            overhead_ratio=overhead_ratio,
            overhead_min_mb=overhead_min_mb,
        )
        explanation.extend(lines)

        # 5) Driver memory
        if driver_memory_gb_override is not None:
            driver_memory_gb = driver_memory_gb_override
            explanation.append(f"Driver memory (override): {driver_memory_gb} GB")
        else:
            driver_memory_gb, lines = driver_memory_gb(
                data_size_gb,
                workload_type,
                driver_default_gb=self._get("driver_memory_gb_default", DRIVER_MEMORY_GB_DEFAULT),
            )
            explanation.extend(lines)

        # 6) Shuffle partitions
        total_cores = total_executor_cores(num_worker_nodes, execs_per_node, executor_cores)
        shuffle_partitions, lines = shuffle_partitions_for_workload(
            total_cores,
            data_size_gb,
            workload_type,
        )
        explanation.extend(lines)

        # 7) Partitions for write (if target file size given)
        num_partitions_for_write: Optional[int] = None
        if target_file_size_mb is not None:
            n, lines = partitions_for_target_file_size(
                data_size_gb,
                target_file_size_mb,
                min_partitions=self._get("write_partitions_min", WRITE_PARTITIONS_MIN),
                max_partitions=self._get("write_partitions_max", WRITE_PARTITIONS_MAX),
            )
            explanation.extend(lines)
            n, cap_lines = cap_partitions_by_parallelism(n, total_cores)
            explanation.extend(cap_lines)
            num_partitions_for_write = n

        # Build Spark config dict (keys as Spark expects)
        config_dict: dict[str, Any] = {
            "spark.executor.instances": str(num_executors),
            "spark.executor.cores": str(executor_cores),
            "spark.executor.memory": f"{int(executor_memory_gb)}g",
            "spark.executor.memoryOverhead": f"{overhead_mb}m",
            "spark.driver.memory": f"{int(driver_memory_gb)}g",
            "spark.sql.shuffle.partitions": str(shuffle_partitions),
        }
        # Optional: avoid speculative execution for stability (can be overridden by user)
        # config_dict["spark.speculation"] = "false"

        session_config = SparkSessionConfig(
            config_dict=config_dict,
            shuffle_partitions=shuffle_partitions,
            num_partitions_for_write=num_partitions_for_write,
            executor_memory_gb=executor_memory_gb,
            executor_cores=executor_cores,
            num_executors=num_executors,
            driver_memory_gb=driver_memory_gb,
            memory_overhead_mb=overhead_mb,
            explanation=explanation,
        )

        return ClusterRecommendation(
            config=session_config,
            total_executor_cores=total_cores,
            num_executors=num_executors,
            safety_notes=safety_notes,
        )

    def recommend_from_cluster_config(
        self,
        cluster_config: ClusterConfig,
        workload_type: WorkloadType,
        data_size_gb: float,
        *,
        target_file_size_mb: Optional[float] = None,
    ) -> ClusterRecommendation:
        """
        Same as recommend() but takes a ClusterConfig and respects overrides
        (executor_cores, executor_memory_gb, driver_memory_gb).
        """
        return self.recommend(
            node_memory_gb=cluster_config.node_memory_gb,
            node_cores=cluster_config.node_cores,
            num_worker_nodes=cluster_config.num_worker_nodes,
            workload_type=workload_type,
            data_size_gb=data_size_gb,
            target_file_size_mb=target_file_size_mb,
            executor_cores_override=cluster_config.executor_cores,
            executor_memory_gb_override=cluster_config.executor_memory_gb,
            driver_memory_gb_override=cluster_config.driver_memory_gb,
        )
