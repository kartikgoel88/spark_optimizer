"""
Structured logging for Spark optimizer.

Provides a simple key=value style for log messages so that log aggregators
can parse and index optimization decisions. Use configure_spark_optimizer_logging()
at application startup if desired.
"""

import json
import logging
import sys
from typing import Any


class StructuredFormatter(logging.Formatter):
    """
    Format log records as JSON or key=value for structured log aggregation.
    """

    def __init__(self, use_json: bool = True):
        super().__init__()
        self._use_json = use_json

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "message": record.getMessage(),
            "level": record.levelname,
            "logger": record.name,
        }
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        if self._use_json:
            return json.dumps(base, default=str)
        parts = [f"level={record.levelname}", f"logger={record.name}", f"msg={record.getMessage()}"]
        return " ".join(parts)


def configure_spark_optimizer_logging(
    level: int = logging.INFO,
    use_json: bool = False,
    stream: Any = None,
) -> None:
    """
    Configure the spark_optimizer logger with structured formatting.

    Args:
        level: Logging level (default INFO).
        use_json: If True, emit JSON lines; if False, emit key=value style.
        stream: Output stream (default sys.stderr).
    """
    stream = stream or sys.stderr
    logger = logging.getLogger("spark_optimizer")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter(use_json=use_json))
        logger.addHandler(handler)
