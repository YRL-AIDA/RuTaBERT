from datetime import datetime
from typing import Optional

import os


class Logger:
    """
    TODO
    """

    def __init__(self, filename: str = "train.log"):
        self.filename = filename

    @staticmethod
    def nvidia_smi():
        os.system("echo nvidia-smi >> logs/gpu.log")

    def info(self, msg: str, level: Optional[str] = None):
        self._log("INFO", msg, level)

    def warn(self, msg: str, level: Optional[str] = None):
        self._log("WARN", msg, level)

    def error(self, msg: str, level: Optional[str] = None):
        self._log("ERROR", msg, level)

    def _log(self, tag: str, msg: str, level: Optional[str] = None):
        msg = (
            f"{datetime.now():%d/%m/%y %H:%M:%S} [{tag}] [{level}] {msg}\n"
        )
        with open(self.filename, mode="a") as f:
            f.write(msg)


if __name__ == "__main__":
    logger = Logger()

    logger.info("epoch started", "epoch")
