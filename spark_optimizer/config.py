"""
Configuration models for Spark optimizer.

All inputs and limits are config-driven; no hardcoded magic numbers in core logic.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class StorageType(str, Enum):
    """Supported storage backends for write optimization."""

    S3 = "s3"
    ADLS = "adls"
    HDFS = "hdfs"
    LOCAL = "local"


class WorkloadType(str, Enum):
    """Workload profile affecting executor and parallelism choices."""

    ETL = "etl"  # Scan + filter + write; CPU-bound, moderate shuffle
    HEAVY_SHUFFLE = "heavy_shuffle"  # Joins/aggregations; shuffle-heavy
    ML = "ml"  # ML pipelines; memory and CPU
    STREAMING = "streaming"  # Micro-batches; lower parallelism, stable memory


class ClusterConfig(BaseModel):
    """Cluster topology and resources used for tuning."""

    num_worker_nodes: int = Field(..., ge=1, description="Number of worker nodes")
    node_memory_gb: float = Field(..., gt=0, description="Memory per node in GB")
    node_cores: int = Field(..., ge=1, description="CPU cores per node")
    num_executors_per_node: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override executors per node (otherwise computed)",
    )
    executor_cores: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override cores per executor (otherwise computed)",
    )
    executor_memory_gb: Optional[float] = Field(
        default=None,
        gt=0,
        description="Override executor memory in GB (otherwise computed)",
    )
    driver_memory_gb: Optional[float] = Field(
        default=None,
        gt=0,
        description="Override driver memory in GB (otherwise computed)",
    )
    memory_overhead_ratio: float = Field(
        default=0.10,
        ge=0.05,
        le=0.40,
        description="Fraction of executor memory for off-heap/overhead (Spark 3.x default ~0.10)",
    )
    memory_overhead_min_mb: int = Field(
        default=384,
        ge=256,
        description="Minimum memory overhead in MB (Spark default 384)",
    )

    @property
    def total_cores(self) -> int:
        return self.num_worker_nodes * self.node_cores

    @property
    def total_memory_gb(self) -> float:
        return self.num_worker_nodes * self.node_memory_gb


class WriteOptimizationConfig(BaseModel):
    """Inputs for optimizing a DataFrame before write."""

    estimated_size_gb: float = Field(..., gt=0, description="Estimated dataset size in GB")
    target_file_size_mb: float = Field(
        default=128.0,
        ge=64.0,
        le=512.0,
        description="Target size per output file in MB (128–256 typical)",
    )
    storage_type: StorageType = Field(
        default=StorageType.S3,
        description="Storage backend (affects block size hints)",
    )
    cluster_config: Optional[ClusterConfig] = Field(
        default=None,
        description="If provided, partition count is capped by cluster parallelism",
    )
    max_partitions: Optional[int] = Field(
        default=None,
        ge=1,
        description="Hard cap on number of partitions before write",
    )
    min_partitions: int = Field(
        default=1,
        ge=1,
        description="Minimum partitions (avoid single huge partition)",
    )
    prefer_coalesce: bool = Field(
        default=False,
        description="Use coalesce instead of repartition when reducing partitions (narrow, no shuffle)",
    )

    @field_validator("target_file_size_mb")
    @classmethod
    def clamp_target_file_size(cls, v: float) -> float:
        return max(64.0, min(512.0, v))


class SparkSessionConfig(BaseModel):
    """Recommended Spark configuration (output of calculator / tuner)."""

    config_dict: dict[str, Any] = Field(
        default_factory=dict,
        description="Spark config key-value pairs to apply",
    )
    shuffle_partitions: Optional[int] = Field(
        default=None,
        description="Recommended spark.sql.shuffle.partitions",
    )
    num_partitions_for_write: Optional[int] = Field(
        default=None,
        description="Recommended partition count before write",
    )
    executor_memory_gb: Optional[float] = None
    executor_cores: Optional[int] = None
    num_executors: Optional[int] = None
    driver_memory_gb: Optional[float] = None
    memory_overhead_mb: Optional[int] = None
    explanation: list[str] = Field(
        default_factory=list,
        description="Human-readable explanation of calculations",
    )
