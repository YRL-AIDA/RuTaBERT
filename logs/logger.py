from datetime import datetime
from typing import Optional

import os


class Logger:
    """Filesystem logger.

    Log info into log files with timestamp. Supports log levels, tags and messages.

    Args:
        filename: Filename of the log file.
    """
    def __init__(self, filename: str = "train.log"):
        self.filename = filename

    @staticmethod
    def nvidia_smi() -> None:
        """Log NVIDIA-smi into log file.

        Warnings:
            If training started on a machine without NVIDIA GPUs, warns will be thrown into error log.

        Returns:
            None

        Todo:
            * If training started on a machine without NVIDIA GPUs warnings will be thrown.
        """
        os.system("nvidia-smi >> logs/gpu.log")

    def info(self, msg: str, level: Optional[str] = None) -> None:
        """Log info message.

        Args:
            msg: Log message.
            level: Log level.

        Returns:
            None
        """
        self._log("INFO", msg, level)

    def warn(self, msg: str, level: Optional[str] = None) -> None:
        """Log warning message.

        Args:
            msg: Log message.
            level: Log level.

        Returns:
            None
        """
        self._log("WARN", msg, level)

    def error(self, msg: str, level: Optional[str] = None) -> None:
        """Log error message.

        Args:
            msg: Log message.
            level: Log level.

        Returns:
            None
        """
        self._log("ERROR", msg, level)

    def _log(self, tag: str, msg: str, level: Optional[str] = None) -> None:
        """Log message into file.

        Log format: [dd/mm/yy H:M:S] [tag] [level] [msg]

        Args:
            tag: Log tag, could be [INFO, WARN, ERROR].
            msg: Log message.
            level: Log level.

        Returns:
            None
        """
        msg = f"{datetime.now():%d/%m/%y %H:%M:%S} [{tag}] [{level}] {msg}\n"
        with open(self.filename, mode="a") as f:
            f.write(msg)


if __name__ == "__main__":
    logger = Logger()
    logger.info("epoch started", "epoch")
