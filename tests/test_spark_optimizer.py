"""
Tests for Spark optimizer (formulas, cluster calculator, config).

No PySpark required; tests use pure Python and Pydantic models.
"""

import pytest

from spark_optimizer.config import (
    ClusterConfig,
    StorageType,
    WorkloadType,
    WriteOptimizationConfig,
)
from spark_optimizer.constants import (
    WRITE_PARTITIONS_MAX,
    WRITE_PARTITIONS_MIN,
)
from spark_optimizer.formulas import (
    cap_partitions_by_parallelism,
    executor_cores_per_node,
    executor_memory_gb_per_node,
    executors_per_node,
    memory_overhead_mb,
    partitions_for_target_file_size,
    shuffle_partitions_for_workload,
    total_executor_cores,
)
from spark_optimizer.cluster_calculator import ClusterCalculator


# --- Config ---


def test_cluster_config_total_cores_and_memory() -> None:
    c = ClusterConfig(num_worker_nodes=10, node_memory_gb=64, node_cores=16)
    assert c.total_cores == 160
    assert c.total_memory_gb == 640


def test_write_optimization_config_target_file_size_clamped() -> None:
    w = WriteOptimizationConfig(estimated_size_gb=100, target_file_size_mb=32)
    assert w.target_file_size_mb >= 64.0
    w2 = WriteOptimizationConfig(estimated_size_gb=100, target_file_size_mb=1024)
    assert w2.target_file_size_mb <= 512.0


# --- Formulas: partitions for file size ---


def test_partitions_for_target_file_size() -> None:
    # 100 GB / 128 MB -> 100*1024/128 = 800
    n, expl = partitions_for_target_file_size(100, 128)
    assert n == 800
    assert any("800" in e for e in expl)

    n2, _ = partitions_for_target_file_size(0.5, 128, min_partitions=4, max_partitions=10)
    assert n2 == 4  # ceil(4) = 4, but min_partitions=4

    n3, _ = partitions_for_target_file_size(1000, 64, max_partitions=5000)
    assert n3 == 5000  # capped


def test_cap_partitions_by_parallelism() -> None:
    n, expl = cap_partitions_by_parallelism(5000, total_executor_cores=200, multiplier_max=4)
    assert n == 800  # 200*4
    n2, _ = cap_partitions_by_parallelism(100, total_executor_cores=200)
    assert n2 == 100  # no cap needed


# --- Formulas: shuffle ---


def test_shuffle_partitions_for_workload() -> None:
    n_etl, _ = shuffle_partitions_for_workload(100, 50, WorkloadType.ETL)
    assert n_etl == 200  # 2x cores
    n_heavy, _ = shuffle_partitions_for_workload(100, 50, WorkloadType.HEAVY_SHUFFLE)
    assert n_heavy == 400  # 4x cores
    n_stream, _ = shuffle_partitions_for_workload(100, 10, WorkloadType.STREAMING)
    assert 100 <= n_stream <= 200  # 1.5x


# --- Formulas: executor and memory ---


def test_executor_cores_per_node() -> None:
    c, _ = executor_cores_per_node(16, cores_reserved=1, preferred=5)
    assert c == 5
    c2, _ = executor_cores_per_node(4, cores_reserved=1, preferred=5)
    assert c2 == 3  # min(5, 3, 8) = 3, then max(1, 3) = 3


def test_executors_per_node() -> None:
    n, _ = executors_per_node(16, 5, cores_reserved=1)
    assert n == (16 - 1) // 5  # 3


def test_executor_memory_gb_per_node() -> None:
    mem, _ = executor_memory_gb_per_node(64, 3, memory_reserved_gb=1)
    assert abs(mem - (63 / 3)) < 0.01
    mem2, _ = executor_memory_gb_per_node(64, 3, memory_max_gb=10)
    assert mem2 == 10  # clamped


def test_memory_overhead_mb() -> None:
    mb, _ = memory_overhead_mb(10, 0.10, 384)
    assert mb == max(1024, 384)  # 10*1024*0.1 = 1024
    mb2, _ = memory_overhead_mb(1, 0.10, 384)
    assert mb2 == 384  # min


def test_total_executor_cores() -> None:
    assert total_executor_cores(10, 3, 5) == 150


# --- Cluster calculator ---


def test_cluster_calculator_recommend() -> None:
    calc = ClusterCalculator()
    rec = calc.recommend(
        node_memory_gb=64,
        node_cores=16,
        num_worker_nodes=10,
        workload_type=WorkloadType.ETL,
        data_size_gb=100,
        target_file_size_mb=128,
    )
    assert rec.num_executors > 0
    assert rec.total_executor_cores > 0
    assert rec.config.shuffle_partitions is not None
    assert rec.config.num_partitions_for_write is not None
    assert "spark.executor.instances" in rec.to_spark_config_dict()
    assert len(rec.config.explanation) > 0


def test_cluster_calculator_from_cluster_config() -> None:
    calc = ClusterCalculator()
    cluster = ClusterConfig(num_worker_nodes=5, node_memory_gb=32, node_cores=8)
    rec = calc.recommend_from_cluster_config(
        cluster,
        WorkloadType.HEAVY_SHUFFLE,
        data_size_gb=50,
        target_file_size_mb=256,
    )
    assert rec.config.executor_cores is not None
    assert rec.config.executor_memory_gb is not None
    assert rec.config.driver_memory_gb is not None


def test_cluster_calculator_respects_overrides() -> None:
    cluster = ClusterConfig(
        num_worker_nodes=4,
        node_memory_gb=64,
        node_cores=16,
        executor_cores=4,
        executor_memory_gb=8,
        driver_memory_gb=4,
    )
    calc = ClusterCalculator()
    rec = calc.recommend_from_cluster_config(
        cluster,
        WorkloadType.ETL,
        data_size_gb=100,
    )
    assert rec.config.executor_cores == 4
    assert rec.config.executor_memory_gb == 8
    assert rec.config.driver_memory_gb == 4
