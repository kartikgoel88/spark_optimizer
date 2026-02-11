# Spark Optimizer

Production-grade, config-driven PySpark 3.x optimization for **large datasets (100GB+)**. Automatically computes optimal executors, memory, shuffle partitions, and write partition count; applies repartition/coalesce to avoid the small-file problem.

## Features

- **Dynamic configuration**: Optimal `spark.sql.shuffle.partitions`, write partition count, executor memory/cores, driver memory, memory overhead.
- **Config-driven**: No hardcoded values; all limits and defaults come from config or `constants`.
- **Safety guards**: Caps to prevent over-allocation and GC-heavy settings (e.g. executor memory ≤ 64GB, 4–5 cores per executor).
- **Extensible**: Works with Databricks, EMR, or any Spark 3.x cluster; override any value via `ClusterConfig` or limits.
- **Structured logging**: Optional JSON or key=value logging for optimization decisions.

## Install

```bash
pip install spark-optimizer
# With PySpark for optimize_for_write and tune_spark_session:
pip install spark-optimizer[spark]
```

## Quick start

### Get recommended config (no Spark session)

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
```

### Tune SparkSession at job start

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

### Optimize DataFrame before write

```python
from spark_optimizer import SparkOptimizer, WriteOptimizationConfig, StorageType, ClusterConfig

df = spark.read.parquet("s3a://bucket/input/")
cluster = ClusterConfig(num_worker_nodes=10, node_memory_gb=64, node_cores=16)
write_config = WriteOptimizationConfig(
    estimated_size_gb=150,
    target_file_size_mb=128,
    storage_type=StorageType.S3,
    cluster_config=cluster,
)
optimizer = SparkOptimizer()
df_out = optimizer.optimize_for_write(df, write_config)
df_out.write.mode("overwrite").parquet("s3a://bucket/output/")
```

## Module layout

```
spark_optimizer/
  __init__.py          # Public API
  config.py            # Pydantic: ClusterConfig, WriteOptimizationConfig, WorkloadType, StorageType
  constants.py         # Safety limits and defaults (overridable)
  formulas.py          # Pure functions with documented formulas
  cluster_calculator.py # ClusterRecommendation from node/workload/data size
  optimizer.py         # SparkOptimizer: optimize_for_write(), tune_spark_session()
  logging_config.py    # Structured logging setup
```

## Example batch job

```bash
# With PySpark installed
DATA_SIZE_GB=150 TARGET_FILE_MB=128 WORKLOAD_TYPE=heavy_shuffle python examples/spark_optimizer_batch_job.py
```

Or with `spark-submit`:

```bash
spark-submit --master 'local[*]' examples/spark_optimizer_batch_job.py
```

## Tests

No PySpark required for tests (they use pure Python and Pydantic):

```bash
pip install -e ".[dev]"
pytest tests/
```

## Dependencies

- **Required**: `pydantic`
- **Optional**: `pyspark` for `optimize_for_write` and `tune_spark_session`. Install with `pip install spark-optimizer[spark]`.
