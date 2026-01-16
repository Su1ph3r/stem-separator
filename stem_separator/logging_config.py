"""
Logging configuration for Stem Separator.

Provides configurable logging with verbose/quiet modes and rich formatting.
"""

from __future__ import annotations

import logging
import sys
from enum import IntEnum
from typing import Optional

# Try to import rich for pretty formatting
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class LogLevel(IntEnum):
    """Log level enumeration."""

    QUIET = logging.WARNING
    NORMAL = logging.INFO
    VERBOSE = logging.DEBUG


# Global logger instance
_logger: Optional[logging.Logger] = None
_console: Optional["Console"] = None
_log_level: LogLevel = LogLevel.NORMAL


def get_console() -> Optional["Console"]:
    """Get the rich console instance if available."""
    global _console
    if RICH_AVAILABLE and _console is None:
        _console = Console()
    return _console


def setup_logging(
    verbose: bool = False,
    quiet: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        verbose: Enable verbose (debug) output.
        quiet: Enable quiet mode (warnings only).
        log_file: Optional file path to write logs to.

    Returns:
        Configured logger instance.
    """
    global _logger, _log_level

    if verbose and quiet:
        quiet = False  # Verbose takes precedence

    if quiet:
        _log_level = LogLevel.QUIET
    elif verbose:
        _log_level = LogLevel.VERBOSE
    else:
        _log_level = LogLevel.NORMAL

    # Create logger
    logger = logging.getLogger("stem_separator")
    logger.setLevel(logging.DEBUG)  # Capture all levels
    logger.handlers.clear()

    # Console handler
    if RICH_AVAILABLE and not quiet:
        console_handler = RichHandler(
            console=get_console(),
            show_time=verbose,
            show_path=verbose,
            rich_tracebacks=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "%(message)s" if not verbose else "%(asctime)s - %(levelname)s - %(message)s"
            )
        )

    console_handler.setLevel(_log_level)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        )
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the configured logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


def is_quiet() -> bool:
    """Check if quiet mode is enabled."""
    return _log_level == LogLevel.QUIET


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return _log_level == LogLevel.VERBOSE


def create_progress(
    description: str = "Processing",
    total: Optional[float] = None,
    transient: bool = False,
) -> "Progress":
    """
    Create a progress bar.

    Args:
        description: Description text for the progress bar.
        total: Total number of steps (None for indeterminate).
        transient: Whether to remove the progress bar when done.

    Returns:
        Progress context manager.
    """
    if not RICH_AVAILABLE or is_quiet():
        # Return a dummy progress that does nothing
        return DummyProgress()

    columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ]

    if total is not None:
        columns.insert(-1, TimeRemainingColumn())

    return Progress(
        *columns,
        console=get_console(),
        transient=transient,
    )


class DummyProgress:
    """Dummy progress bar for quiet mode or when rich is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add_task(self, description: str, total: Optional[float] = None, **kwargs) -> int:
        return 0

    def update(self, task_id: int, **kwargs):
        pass

    def advance(self, task_id: int, advance: float = 1):
        pass

    def remove_task(self, task_id: int):
        pass


class DummyTaskID:
    """Dummy task ID for progress tracking."""

    pass


def print_status(message: str, style: str = ""):
    """
    Print a status message.

    Args:
        message: Message to print.
        style: Rich style string (ignored if rich unavailable).
    """
    if is_quiet():
        return

    console = get_console()
    if console and RICH_AVAILABLE:
        console.print(message, style=style)
    else:
        print(message)


def print_error(message: str):
    """Print an error message."""
    console = get_console()
    if console and RICH_AVAILABLE:
        console.print(f"[bold red]Error:[/bold red] {message}")
    else:
        print(f"Error: {message}", file=sys.stderr)


def print_warning(message: str):
    """Print a warning message."""
    console = get_console()
    if console and RICH_AVAILABLE:
        console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
    else:
        print(f"Warning: {message}", file=sys.stderr)


def print_success(message: str):
    """Print a success message."""
    if is_quiet():
        return

    console = get_console()
    if console and RICH_AVAILABLE:
        console.print(f"[bold green]{message}[/bold green]")
    else:
        print(message)


def print_info(message: str):
    """Print an info message."""
    if is_quiet():
        return

    console = get_console()
    if console and RICH_AVAILABLE:
        console.print(f"[dim]{message}[/dim]")
    else:
        print(message)
