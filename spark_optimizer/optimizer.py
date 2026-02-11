"""
SparkOptimizer: tune SparkSession and optimize DataFrames for write.

Uses ClusterCalculator and formulas to apply config-driven optimizations
and structured logging.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from spark_optimizer.cluster_calculator import (
    ClusterCalculator,
    ClusterRecommendation,
)
from spark_optimizer.config import (
    ClusterConfig,
    StorageType,
    WorkloadType,
    WriteOptimizationConfig,
)
from spark_optimizer.formulas import (
    cap_partitions_by_parallelism,
    partitions_for_target_file_size,
)
from spark_optimizer.constants import WRITE_PARTITIONS_MAX

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def _get_spark_logger() -> logging.Logger:
    return logging.getLogger("spark_optimizer")


class SparkOptimizer:
    """
    Production-grade Spark 3.x optimization utility.

    - optimize_for_write(df, config): repartition/coalesce DataFrame for write; avoid small files.
    - tune_spark_session(spark, cluster_config, ...): apply recommended Spark config to session.
    """

    def __init__(
        self,
        cluster_calculator: Optional[ClusterCalculator] = None,
        limits: Optional[dict[str, Any]] = None,
    ):
        self._calculator = cluster_calculator or ClusterCalculator(limits=limits)
        self._log = _get_spark_logger()

    def optimize_for_write(
        self,
        df: "DataFrame",
        config: WriteOptimizationConfig,
    ) -> "DataFrame":
        """
        Optimize a DataFrame for writing: set partition count to hit target file size
        and avoid small-file problem.

        - Computes optimal partition count from estimated_size_gb and target_file_size_mb.
        - Optionally caps by cluster parallelism if cluster_config is provided.
        - Applies repartition(num) or coalesce(num) (when prefer_coalesce and reducing).
        - Logs all decisions.

        Args:
            df: Spark DataFrame to optimize.
            config: Write optimization inputs (size, target file size, storage, cluster, etc.).

        Returns:
            DataFrame with partition count set for write (caller then writes with df.write.save(...)).
        """
        estimated_gb = config.estimated_size_gb
        target_mb = config.target_file_size_mb
        min_p = config.min_partitions
        max_p = config.max_partitions or WRITE_PARTITIONS_MAX

        n, expl = partitions_for_target_file_size(
            estimated_gb,
            target_mb,
            min_partitions=min_p,
            max_partitions=max_p,
        )
        for line in expl:
            self._log.info("write_optimizer: %s", line)

        if config.cluster_config is not None:
            total_cores = self._total_cores_from_cluster(config.cluster_config)
            n, cap_expl = cap_partitions_by_parallelism(n, total_cores)
            for line in cap_expl:
                self._log.info("write_optimizer: %s", line)

        n = max(config.min_partitions, min(n, max_p))

        current_partitions = df.rdd.getNumPartitions()
        self._log.info(
            "write_optimizer: recommended_partitions=%s current_partitions=%s",
            n,
            current_partitions,
        )

        if n == current_partitions:
            self._log.info("write_optimizer: no repartition/coalesce needed")
            return df

        if n < current_partitions and config.prefer_coalesce:
            out = df.coalesce(n)
            self._log.info("write_optimizer: applied coalesce(%s) to reduce partitions", n)
        else:
            out = df.repartition(n)
            self._log.info("write_optimizer: applied repartition(%s)", n)

        return out

    def _total_cores_from_cluster(self, cluster_config: ClusterConfig) -> int:
        """Derive total executor cores from ClusterConfig (for capping partitions)."""
        from spark_optimizer.formulas import (
            executor_cores_per_node,
            executors_per_node,
            total_executor_cores,
        )
        from spark_optimizer.constants import CORES_RESERVED_PER_NODE

        executor_cores = cluster_config.executor_cores
        if executor_cores is None:
            executor_cores, _ = executor_cores_per_node(
                cluster_config.node_cores,
                cores_reserved=CORES_RESERVED_PER_NODE,
            )
        execs_per_node = cluster_config.num_executors_per_node
        if execs_per_node is None:
            execs_per_node, _ = executors_per_node(
                cluster_config.node_cores,
                executor_cores,
                cores_reserved=CORES_RESERVED_PER_NODE,
            )
        return total_executor_cores(
            cluster_config.num_worker_nodes,
            execs_per_node,
            executor_cores,
        )

    def tune_spark_session(
        self,
        spark: "SparkSession",
        cluster_config: Optional[ClusterConfig] = None,
        *,
        workload_type: WorkloadType = WorkloadType.ETL,
        data_size_gb: float = 10.0,
        target_file_size_mb: Optional[float] = None,
        recommendation: Optional[ClusterRecommendation] = None,
        apply_config: bool = True,
    ) -> ClusterRecommendation:
        """
        Tune the Spark session with recommended config (shuffle partitions, executor
        memory hints, etc.). Either pass a precomputed recommendation or let the
        optimizer compute one from cluster_config + workload + data_size_gb.

        Args:
            spark: Active SparkSession.
            cluster_config: Cluster topology; required if recommendation is None.
            workload_type: ETL | heavy_shuffle | ML | streaming.
            data_size_gb: Estimated data size in GB.
            target_file_size_mb: Optional; if set, recommendation includes num_partitions_for_write.
            recommendation: If provided, use this instead of computing from cluster_config.
            apply_config: If True, set spark.conf entries; if False, only return recommendation.

        Returns:
            ClusterRecommendation with config, explanation, and safety_notes.
        """
        if recommendation is None:
            if cluster_config is None:
                raise ValueError("Either cluster_config or recommendation must be provided")
            recommendation = self._calculator.recommend_from_cluster_config(
                cluster_config,
                workload_type,
                data_size_gb,
                target_file_size_mb=target_file_size_mb,
            )

        for line in recommendation.config.explanation:
            self._log.info("tune_session: %s", line)
        for note in recommendation.safety_notes:
            self._log.warning("tune_session safety: %s", note)

        if apply_config:
            conf = recommendation.to_spark_config_dict()
            for k, v in conf.items():
                spark.conf.set(k, v)
            self._log.info("tune_session: applied %s config entries", len(conf))

        return recommendation

    def get_recommendation(
        self,
        node_memory_gb: float,
        node_cores: int,
        num_worker_nodes: int,
        workload_type: WorkloadType,
        data_size_gb: float,
        *,
        target_file_size_mb: Optional[float] = None,
        **overrides: Any,
    ) -> ClusterRecommendation:
        """
        Get a ClusterRecommendation without a Spark session (e.g. for job submission config).
        Same inputs as ClusterCalculator.recommend().
        """
        return self._calculator.recommend(
            node_memory_gb=node_memory_gb,
            node_cores=node_cores,
            num_worker_nodes=num_worker_nodes,
            workload_type=workload_type,
            data_size_gb=data_size_gb,
            target_file_size_mb=target_file_size_mb,
            **overrides,
        )
