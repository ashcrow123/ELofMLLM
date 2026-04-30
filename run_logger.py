import logging
import os


def create_run_logger(name: str, log_dir: str = "./log")->logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"run.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
