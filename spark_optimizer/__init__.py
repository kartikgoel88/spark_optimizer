"""
PySpark 3.x optimization utility module.

Provides config-driven, production-grade utilities to:
- Dynamically compute optimal Spark configs (shuffle partitions, executors, memory)
- Optimize DataFrames for write (repartition/coalesce, target file size)
- Tune SparkSession for cluster and workload
- Avoid small-file problem and GC-heavy configurations

Use with PySpark 3.x on Databricks, EMR, or standalone clusters.
"""

from spark_optimizer.config import (
    ClusterConfig,
    StorageType,
    WorkloadType,
    WriteOptimizationConfig,
    SparkSessionConfig,
)
from spark_optimizer.cluster_calculator import (
    ClusterCalculator,
    ClusterRecommendation,
)
from spark_optimizer.optimizer import SparkOptimizer

__all__ = [
    "SparkOptimizer",
    "ClusterCalculator",
    "ClusterRecommendation",
    "ClusterConfig",
    "WriteOptimizationConfig",
    "SparkSessionConfig",
    "StorageType",
    "WorkloadType",
]
