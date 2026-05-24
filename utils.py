"""utils.py - Shared utilities."""
import logging
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("forecast")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
