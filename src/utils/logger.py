import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured application logger.
    """

    # Create logs directory
    log_directory = Path("logs")
    log_directory.mkdir(parents=True, exist_ok=True)

    log_file = log_directory / "application.log"

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger