#!/usr/bin/env python3
"""
CA7 Excel metadata to Airflow DAG YAML converter.

Reads an Excel file with job_name, dependencies, and jcl_name columns,
normalizes CA7 naming (strip last 2 chars), merges by logical job and module,
validates the dependency graph, and writes one YAML file per module.

- Missing upstream jobs are emitted as ExternalTaskSensor tasks.
- Dependencies that start with "/" are treated as mutually exclusive (stored
  in mutually_exclusive_with in the task YAML).
- Dependencies that start with "J" are time dependencies; the last 4 characters
  denote the time (HHMM). Emitted as TimeSensor tasks.

Dependencies: pip install pandas openpyxl pyyaml
"""

import argparse
import logging
import sys
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CA7 Naming: normalize by removing last 2 characters (day + run)
# ---------------------------------------------------------------------------
def normalize_job_name(name: str) -> str:
    """Normalize CA7 job name by removing last 2 chars (e.g. HQEAED21 -> HQEAED)."""
    if not name or not isinstance(name, str):
        return ""
    s = str(name).strip()
    return s[:-2] if len(s) >= 2 else s


def get_module(normalized_name: str) -> str:
    """Module = first 3 characters of normalized job name."""
    if not normalized_name or len(normalized_name) < 3:
        return normalized_name
    return normalized_name[:3]


# ---------------------------------------------------------------------------
# Load and normalize data
# ---------------------------------------------------------------------------
def load_excel(path: str) -> pd.DataFrame:
    """Load Excel file; expect columns: job_name, dependencies, jcl_name."""
    logger.info("Loading Excel: %s", path)
    df = pd.read_excel(path)
    required = {"job_name", "dependencies", "jcl_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected: {required}")
    return df


def build_logical_jobs_and_deps(df: pd.DataFrame):
    """
    Normalize all job names and dependencies, merge duplicates.
    Dependency conventions:
      - "/" prefix => mutually exclusive
      - "J" prefix => time dependency; last 4 chars = time (HHMM)
      - else => normal upstream job dependency
    Returns:
        logical_jobs, deps, mutually_exclusive_deps, time_deps
        time_deps: dict[normalized_name] -> set of "HHMM" strings
    """
    logical_jobs = {}  # normalized_name -> { jcl_name, module }
    deps = defaultdict(set)  # normalized_name -> set of upstream normalized names
    mutually_exclusive_deps = defaultdict(set)  # "/" prefix = mutually exclusive
    time_deps = defaultdict(set)  # "J" prefix => last 4 chars = time (HHMM)

    for _, row in df.iterrows():
        raw_name = row["job_name"]
        raw_deps = row["dependencies"]
        jcl_name = row["jcl_name"]

        if pd.isna(raw_name):
            continue

        norm = normalize_job_name(str(raw_name))
        if not norm:
            continue

        # Merge: keep one jcl_name per logical job (last seen wins)
        if pd.notna(jcl_name) and str(jcl_name).strip():
            logical_jobs[norm] = {
                "jcl_name": str(jcl_name).strip(),
                "module": get_module(norm),
            }

        # Parse dependencies: "/" => mutually exclusive, "J" => time (last 4 chars), else normal
        if pd.notna(raw_deps) and str(raw_deps).strip():
            for d in str(raw_deps).split(","):
                d = d.strip()
                if not d:
                    continue
                if d.startswith("/"):
                    dep_norm = normalize_job_name(d[1:].strip())
                    if dep_norm:
                        mutually_exclusive_deps[norm].add(dep_norm)
                elif d.startswith("J") and len(d) >= 5:
                    # Time dependency: last 4 characters denote time (HHMM)
                    time_deps[norm].add(d[-4:])
                else:
                    dep_norm = normalize_job_name(d)
                    if dep_norm:
                        deps[norm].add(dep_norm)

    return dict(logical_jobs), dict(deps), dict(mutually_exclusive_deps), dict(time_deps)


# ---------------------------------------------------------------------------
# Topological sort and validation
# ---------------------------------------------------------------------------
def topological_sort(nodes: list[str], deps: dict[str, set]) -> list[str]:
    """
    Kahn's algorithm. Returns ordered list of task_ids.
    Raises if circular dependency detected.
    Secondary ordering at each level: alphabetical.
    """
    node_set = set(nodes)
    # in_degree[n] = number of upstream deps that are in this module
    in_degree = {n: 0 for n in nodes}
    for n in nodes:
        for u in deps.get(n, set()):
            if u in node_set:
                in_degree[n] += 1

    # Reverse graph: rev[u] = list of n such that u in deps[n] (n depends on u)
    rev = defaultdict(list)
    for n in nodes:
        for u in deps.get(n, set()):
            if u in node_set:
                rev[u].append(n)

    q = deque([n for n in nodes if in_degree[n] == 0])
    result = []

    while q:
        # Secondary ordering: alphabetical among same level
        level = sorted(q)
        q.clear()
        for n in level:
            result.append(n)
            for m in rev[n]:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    q.append(m)

    if len(result) != len(nodes):
        raise Exception("Circular dependency detected")

    return result


def validate_and_order(
    logical_jobs: dict, deps: dict, module: str, module_jobs: list[str]
) -> list[str]:
    """
    Validate only in-module deps for circular; missing upstreams are allowed
    (emitted as external sensors). Returns topologically ordered task list.
    """
    return topological_sort(module_jobs, deps)


# ---------------------------------------------------------------------------
# Build and write YAML per module
# ---------------------------------------------------------------------------
def _sensor_task_id(external_task_id: str) -> str:
    """Task id for the ExternalTaskSensor that waits for an external job."""
    return f"wait_for_{external_task_id}"


def _time_sensor_task_id(hhmm: str) -> str:
    """Task id for the TimeSensor that waits until a time of day (HHMM)."""
    return f"time_{hhmm}"


def _hhmm_to_target_time(hhmm: str) -> str:
    """Convert 4-char HHMM to Airflow target_time string 'HH:MM:00'."""
    if len(hhmm) != 4 or not hhmm.isdigit():
        return "00:00:00"
    return f"{hhmm[:2]}:{hhmm[2:]}:00"


def build_dag_yaml(
    module: str,
    ordered_tasks: list[str],
    logical_jobs: dict,
    deps: dict,
    mutually_exclusive_deps: dict,
    time_deps: dict,
) -> dict:
    """
    Build the DAG structure for one module.
    Missing upstreams -> ExternalTaskSensor; "/" deps -> mutually_exclusive_with;
    "J" deps (last 4 = time) -> TimeSensor tasks.
    """
    task_list = []
    ordered_set = set(ordered_tasks)
    # Collect external (missing) upstreams referenced by tasks in this module
    external_upstreams = set()
    for task_id in ordered_tasks:
        for u in deps.get(task_id, set()):
            if u not in logical_jobs:
                external_upstreams.add(u)

    # Sensor tasks for missing upstream jobs (external dependencies)
    for ext_id in sorted(external_upstreams):
        ext_module = get_module(ext_id)
        sensor_id = _sensor_task_id(ext_id)
        task_list.append({
            "task_id": sensor_id,
            "operator": "ExternalTaskSensor",
            "external_dag_id": f"{ext_module}_dag",
            "external_task_id": ext_id,
            "depends_on": [],
        })

    # TimeSensor tasks for time dependencies (J...HHMM)
    time_upstreams = set()
    for task_id in ordered_tasks:
        time_upstreams |= time_deps.get(task_id, set())
    for hhmm in sorted(time_upstreams):
        task_list.append({
            "task_id": _time_sensor_task_id(hhmm),
            "operator": "TimeSensor",
            "target_time": _hhmm_to_target_time(hhmm),
            "depends_on": [],
        })

    # Regular tasks (BashOperator)
    for task_id in ordered_tasks:
        info = logical_jobs.get(task_id, {})
        jcl = info.get("jcl_name", "")
        # In-module upstreams
        upstream = [u for u in deps.get(task_id, set()) if u in ordered_set]
        # External upstreams: depend on the sensor task
        for u in deps.get(task_id, set()):
            if u not in logical_jobs:
                upstream.append(_sensor_task_id(u))
        # Time dependencies: depend on the TimeSensor task
        for hhmm in time_deps.get(task_id, set()):
            upstream.append(_time_sensor_task_id(hhmm))
        upstream = sorted(upstream)
        # Mutually exclusive: only include those that are in this DAG
        mut_ex = sorted(mutually_exclusive_deps.get(task_id, set()) & ordered_set)
        task_def = {
            "task_id": task_id,
            "operator": "BashOperator",
            "bash_command": "{{ params.jcl }}",
            "params": {"jcl": jcl},
            "depends_on": upstream,
        }
        if mut_ex:
            task_def["mutually_exclusive_with"] = mut_ex
        task_list.append(task_def)

    return {
        "dag": {
            "dag_id": f"{module}_dag",
            "schedule": "@daily",
            "default_args": {
                "owner": "ca7_migration",
                "retries": 1,
                "retry_delay_minutes": 5,
            },
            "tasks": task_list,
        }
    }


def write_yaml(path: Path, data: dict) -> None:
    """Write dict to YAML file with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(input_xlsx: str, output_dir: str) -> None:
    """Load Excel, normalize, group by module, validate (circular only), and write YAMLs."""
    df = load_excel(input_xlsx)
    logical_jobs, deps, mutually_exclusive_deps, time_deps = build_logical_jobs_and_deps(df)

    # Group by module
    by_module = defaultdict(list)
    for norm, info in logical_jobs.items():
        mod = info["module"]
        if norm not in by_module[mod]:
            by_module[mod].append(norm)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for module, job_list in by_module.items():
        if not module:
            continue
        ordered = validate_and_order(logical_jobs, deps, module, job_list)
        dag_data = build_dag_yaml(
            module, ordered, logical_jobs, deps, mutually_exclusive_deps, time_deps
        )
        write_yaml(out_path / f"{module}_dag.yaml", dag_data)

    logger.info("Done. Wrote %d module DAG(s) to %s", len(by_module), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CA7 Excel metadata to Airflow DAG YAML files."
    )
    parser.add_argument("input_xlsx", help="Path to input Excel file")
    parser.add_argument("output_dir", help="Output directory for YAML files")
    args = parser.parse_args()

    try:
        run(args.input_xlsx, args.output_dir)
    except Exception as e:
        logger.exception("Failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
