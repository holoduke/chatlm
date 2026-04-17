import logging
import sys


class ColorFormatter(logging.Formatter):
    GREY = "\033[90m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"

    LEVEL_COLOR = {
        logging.DEBUG: GREY,
        logging.INFO: CYAN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLOR.get(record.levelno, self.GREY)
        base = super().format(record)
        return f"{self.GREY}{self.formatTime(record, '%H:%M:%S')}{self.RESET} {color}{record.levelname:<5}{self.RESET} {self.MAGENTA}{record.name}{self.RESET} {base}"


def configure() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter("%(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    logging.getLogger("gemma4").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
