"""Setup structed logging."""
import logging
import sys
from typing import List, Any
import structlog
from structlog.types import EventDict
from structlog.typing import Processor, WrappedLogger, EventDict

_JSON = False


def json_logging_enabled() -> bool:
    global _JSON
    return _JSON


def configure_logging(enable_json_logs: bool = False, log_level: str = "INFO"):
    """Configure logging to use structlog and include additional info."""
    global _JSON
    if enable_json_logs:
        _JSON = True

    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")

    shared_processors: List[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.ExtraAdder(),
        _drop_color_message_key,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.PATHNAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.PROCESS,
            }
        ),
    ]

    if enable_json_logs:
        shared_processors.append(_rename_event_key)
        shared_processors.append(_ordered_keys)
        shared_processors.append(structlog.processors.format_exc_info)

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logs_render: Processor = (
        structlog.processors.JSONRenderer()
        if enable_json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    extra_processors: List[Processor] = []
    # extra_processors = [StructureVLLMOutput()]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors + extra_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            logs_render,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    level = log_level.upper()
    root_logger.setLevel(level)

    for _log in ["uvicorn", "uvicorn.error"]:
        logging.getLogger(_log).handlers.clear()
        logging.getLogger(_log).propagate = True

    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False

    def handle_exception(exc_type, exc_value, exc_traceback):
        """Log any uncaught exception instead of letting it be printed by Python.

        (But leave KeyboardInterrupt untouched to allow users to Ctrl+C to stop).
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        root_logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


def _rename_event_key(_, __, event_dict: EventDict) -> EventDict:
    """Rewrite logs to place the event under the message key expected by DataDog."""
    event_dict["message"] = event_dict.pop("event")
    return event_dict


def _ordered_keys(_, __, event_dict: EventDict) -> EventDict:
    """Place timestamp and message first in log events."""
    event_dict["timestamp"] = event_dict.pop("timestamp")
    event_dict["message"] = event_dict.pop("message")
    return event_dict


def _drop_color_message_key(_, __, event_dict: EventDict) -> EventDict:
    """Drop extra messages logged by Uvicorn under the color_message key."""
    event_dict.pop("color_message", None)
    return event_dict


def log_every(state: Any, iterations: int, log_fn: Any, message: str, **kw_args):
    if not hasattr(state, "_log_iterations"):
        state._log_iterations = {}

    if message not in state._log_iterations:
        state._log_iterations[message] = 1

    if state._log_iterations[message] % iterations == 0:
        log_fn(
            message, log_every=True, iteration=state._log_iterations[message], **kw_args
        )
        state._log_iterations[message] = 1
    else:
        state._log_iterations[message] += 1
