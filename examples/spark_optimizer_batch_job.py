"""
Example batch job using SparkOptimizer for large dataset writes.

Demonstrates:
  - Tuning SparkSession from cluster config and workload
  - Optimizing a DataFrame for write (target file size, avoid small files)
  - Getting recommendation without a session (e.g. for job submission)

Run on a cluster with PySpark 3.x installed, or with spark-submit.
"""

from __future__ import annotations

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from spark_optimizer import (
    ClusterConfig,
    ClusterCalculator,
    SparkOptimizer,
    StorageType,
    WorkloadType,
    WriteOptimizationConfig,
)
from spark_optimizer.logging_config import configure_spark_optimizer_logging

# Optional: JSON structured logs for production
# configure_spark_optimizer_logging(level=logging.INFO, use_json=True)


def main() -> None:
    # --- 1) Cluster and workload inputs (e.g. from env or config) ---
    node_memory_gb = float(os.environ.get("SPARK_NODE_MEMORY_GB", "64"))
    node_cores = int(os.environ.get("SPARK_NODE_CORES", "16"))
    num_workers = int(os.environ.get("SPARK_NUM_WORKERS", "10"))
    data_size_gb = float(os.environ.get("DATA_SIZE_GB", "150"))
    target_file_mb = float(os.environ.get("TARGET_FILE_MB", "128"))
    workload = os.environ.get("WORKLOAD_TYPE", "etl").lower()
    workload_type = getattr(WorkloadType, workload.upper(), WorkloadType.ETL)

    cluster_config = ClusterConfig(
        num_worker_nodes=num_workers,
        node_memory_gb=node_memory_gb,
        node_cores=node_cores,
    )

    optimizer = SparkOptimizer()

    # --- 2) Get full recommendation (no Spark session) ---
    recommendation = optimizer.get_recommendation(
        node_memory_gb=node_memory_gb,
        node_cores=node_cores,
        num_worker_nodes=num_workers,
        workload_type=workload_type,
        data_size_gb=data_size_gb,
        target_file_size_mb=target_file_mb,
    )
    print("--- Recommended Spark configuration ---")
    for k, v in recommendation.to_spark_config_dict().items():
        print(f"  {k}={v}")
    print("--- Explanation ---")
    for line in recommendation.config.explanation:
        print(f"  {line}")
    if recommendation.safety_notes:
        print("--- Safety notes ---")
        for note in recommendation.safety_notes:
            print(f"  ! {note}")
    print(f"  num_partitions_for_write = {recommendation.config.num_partitions_for_write}")

    # --- 3) If running inside Spark (e.g. spark-submit), tune session and optimize write ---
    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("spark_optimizer_example")
            .config("spark.sql.adaptive.enabled", "true")  # Spark 3.x AQE
            .getOrCreate()
        )

        optimizer.tune_spark_session(
            spark,
            cluster_config,
            workload_type=workload_type,
            data_size_gb=data_size_gb,
            target_file_size_mb=target_file_mb,
            apply_config=True,
        )

        # Simulate reading a large dataset
        df = spark.range(0, 10_000_000, numSlices=200).selectExpr(
            "id",
            "id % 100 as key",
            "id * 2 as value",
        )

        write_config = WriteOptimizationConfig(
            estimated_size_gb=data_size_gb,
            target_file_size_mb=target_file_mb,
            storage_type=StorageType.S3,
            cluster_config=cluster_config,
            prefer_coalesce=False,
        )
        df_optimized = optimizer.optimize_for_write(df, write_config)

        # Write (example: local path; use s3a:// or abfs:// in production)
        output_path = os.environ.get("OUTPUT_PATH", "/tmp/spark_optimizer_example")
        df_optimized.write.mode("overwrite").parquet(output_path)
        print(f"Wrote optimized DataFrame to {output_path}")

        spark.stop()
    except ImportError:
        print("PySpark not installed; skipping session tune and write example.")


if __name__ == "__main__":
    main()
