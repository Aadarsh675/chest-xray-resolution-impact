import logging
import colorlog
import os


def get_logger(name: str):
    """
    Sets up and returns a logger with colored output.

    Args:
        name (str): Name of the logger (usually `__name__`).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Check if the logging level is specified in the environment, default to DEBUG
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

    # Create a handler for console output with color formatting
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )

    # Create and configure the logger
    logger = colorlog.getLogger(name)
    logger.setLevel(getattr(logging, log_level, logging.DEBUG))  # Default to DEBUG
    logger.addHandler(handler)

    # Avoid duplicate logs by ensuring handlers are not added multiple times
    logger.propagate = False

    return logger
