# Spark 3.x Optimization Utility

Production-grade, config-driven PySpark 3.x optimization for **large datasets (100GB+)**. Automatically computes optimal executors, memory, shuffle partitions, and write partition count; applies repartition/coalesce to avoid the small-file problem.

## Features

- **Dynamic configuration**: Optimal `spark.sql.shuffle.partitions`, write partition count, executor memory/cores, driver memory, memory overhead.
- **Config-driven**: No hardcoded values; all limits and defaults come from config or `constants`.
- **Safety guards**: Caps to prevent over-allocation and GC-heavy settings (e.g. executor memory ≤ 64GB, 4–5 cores per executor).
- **Extensible**: Works with Databricks, EMR, or any Spark 3.x cluster; override any value via `ClusterConfig` or limits.
- **Structured logging**: Optional JSON or key=value logging for optimization decisions.

## Module layout

```
spark_optimizer/
  __init__.py          # Public API
  config.py            # Pydantic models: ClusterConfig, WriteOptimizationConfig, WorkloadType, StorageType
  constants.py         # Safety limits and defaults (overridable)
  formulas.py          # Pure functions with documented formulas
  cluster_calculator.py # ClusterRecommendation from node/workload/data size
  optimizer.py         # SparkOptimizer: optimize_for_write(), tune_spark_session()
  logging_config.py   # Structured logging setup
```

## Formulas (Spark 3.x best practices)

### 1. Partitions for target file size

**Goal**: ~128–256 MB per output file to avoid small-file problem and respect S3/ADLS/HDFS block sizes.

```
num_partitions = ceil(estimated_size_gb * 1024 / target_file_size_mb)
```

Then clamped to `[min_partitions, max_partitions]` (default max 100,000).

### 2. Cap partitions by cluster parallelism

Avoid more partitions than the cluster can run efficiently:

```
cap = total_executor_cores * multiplier_max   # multiplier_max typically 4
if num_partitions > cap then num_partitions = cap
```

So we never recommend more than ~4× total cores partitions for write.

### 3. Shuffle partitions

**Goal**: Balance shuffle parallelism and task overhead.

- **ETL**: `2 × total_executor_cores`
- **Heavy shuffle**: `4 × total_executor_cores` (or scaled by data size for very large datasets)
- **ML**: `2 × total_executor_cores`
- **Streaming**: `1.5 × total_executor_cores`

Clamped to `[1, 10000]` (configurable).

### 4. Executor cores per node

**Goal**: 4–5 cores per executor to reduce GC pressure (Spark 3.x best practice).

```
available_cores = node_cores - cores_reserved_per_node   # reserve 1 for OS/AM
executor_cores = min(preferred=5, available_cores, executor_cores_max=8)
```

### 5. Executors per node

```
executors_per_node = floor((node_cores - 1) / executor_cores)
total_executors = num_worker_nodes * executors_per_node
```

### 6. Executor memory

```
memory_per_executor = (node_memory_gb - memory_reserved_gb) / executors_per_node
```

Clamped to [2 GB, 64 GB]. Above ~32 GB, the calculator adds a safety note about GC.

### 7. Memory overhead

Off-heap / overhead (Spark 3.x):

```
memory_overhead_mb = max(executor_memory_gb * 1024 * 0.10, 384)
```

### 8. Driver memory

Heuristic by workload and data size:

- Default 4 GB; increase for large data or heavy shuffle (e.g. 8–16 GB).
- Clamped to [1 GB, 32 GB].

## Usage

### 1. Get recommended config (no Spark session)

Use when building `spark-submit` or cluster job config:

```python
from spark_optimizer import SparkOptimizer, WorkloadType

optimizer = SparkOptimizer()
rec = optimizer.get_recommendation(
    node_memory_gb=64,
    node_cores=16,
    num_worker_nodes=10,
    workload_type=WorkloadType.HEAVY_SHUFFLE,
    data_size_gb=500,
    target_file_size_mb=128,
)
print(rec.to_spark_config_dict())
for line in rec.config.explanation:
    print(line)
```

### 2. Tune SparkSession at job start

```python
from pyspark.sql import SparkSession
from spark_optimizer import SparkOptimizer, ClusterConfig, WorkloadType

spark = SparkSession.builder.appName("my_job").getOrCreate()
cluster = ClusterConfig(num_worker_nodes=10, node_memory_gb=64, node_cores=16)
optimizer = SparkOptimizer()
optimizer.tune_spark_session(
    spark,
    cluster,
    workload_type=WorkloadType.ETL,
    data_size_gb=200,
    target_file_size_mb=256,
    apply_config=True,
)
```

### 3. Optimize DataFrame before write

```python
from spark_optimizer import SparkOptimizer, WriteOptimizationConfig, StorageType, ClusterConfig

df = spark.read.parquet("s3a://bucket/input/")
cluster = ClusterConfig(num_worker_nodes=10, node_memory_gb=64, node_cores=16)
write_config = WriteOptimizationConfig(
    estimated_size_gb=150,
    target_file_size_mb=128,
    storage_type=StorageType.S3,
    cluster_config=cluster,
    prefer_coalesce=False,  # use repartition for better balance when increasing partitions
)
optimizer = SparkOptimizer()
df_out = optimizer.optimize_for_write(df, write_config)
df_out.write.mode("overwrite").parquet("s3a://bucket/output/")
```

### 4. Using ClusterCalculator directly

```python
from spark_optimizer import ClusterConfig, ClusterCalculator, WorkloadType

calc = ClusterCalculator()
cluster = ClusterConfig(num_worker_nodes=20, node_memory_gb=128, node_cores=32)
rec = calc.recommend_from_cluster_config(
    cluster,
    WorkloadType.ETL,
    data_size_gb=1000,
    target_file_size_mb=256,
)
print(rec.config.num_partitions_for_write)
print(rec.config.shuffle_partitions)
```

## Best practices (Spark 3.x)

1. **Enable AQE**: `spark.sql.adaptive.enabled = true` (default in Spark 3.x).
2. **Executor size**: Prefer 4–5 cores per executor; avoid very large heaps (>32 GB) to reduce GC.
3. **File size**: Target 128–256 MB per file for S3/ADLS/HDFS.
4. **Shuffle**: For shuffle-heavy jobs, use `WorkloadType.HEAVY_SHUFFLE` so shuffle partitions scale.
5. **Overrides**: Use `ClusterConfig(executor_cores=..., executor_memory_gb=..., driver_memory_gb=...)` when cluster policy fixes these.
6. **Databricks/EMR**: Pass cluster dimensions from environment or cluster API; use same formulas. For serverless, use estimated data size and virtual cores.

## Safety guards

- Executor memory clamped to [2, 64] GB.
- Executor cores clamped to [1, 8]; recommended 5.
- Shuffle and write partitions clamped to configured min/max.
- Partition count for write capped at ~4× total executor cores.
- Warning logged if executor memory > 32 GB (GC risk).

## Dependencies

- **Required**: `pydantic`.
- **Optional**: `pyspark` (for `optimize_for_write` and `tune_spark_session`). Install with `pip install spark-optimizer[spark]`.

## Example batch job

See `examples/spark_optimizer_batch_job.py` for a full example that:

- Reads cluster and data size from environment.
- Gets a full recommendation and prints config + explanation.
- If PySpark is available, tunes the session, builds a sample DataFrame, optimizes for write, and writes Parquet.

Run (with PySpark installed):

```bash
DATA_SIZE_GB=150 TARGET_FILE_MB=128 WORKLOAD_TYPE=heavy_shuffle python examples/spark_optimizer_batch_job.py
```

Or with `spark-submit`:

```bash
spark-submit --master 'local[*]' examples/spark_optimizer_batch_job.py
```
