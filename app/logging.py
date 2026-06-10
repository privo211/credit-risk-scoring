"""
Structured logging configuration using structlog.

Call setup_logging(level="INFO") during app initialization to configure.
Use get_logger(__name__) in modules instead of logging.getLogger().
"""

import structlog
import logging
from typing import Optional


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure structlog with either JSON (production) or console (dev) rendering.
    
    Args:
        level: Logging level string (e.g., "INFO", "DEBUG", "WARNING").
        json_format: If True, output JSON lines. If False, use rich console output.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.PositionalArgumentsFormatter(),
    ]

    if json_format:
        # Production: JSON output for log aggregation
        processors = shared_processors + [
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: colorful console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Redirect stdlib logging to structlog
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger bound with the module name."""
    return structlog.get_logger(name or __name__)
