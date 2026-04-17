import logging

def get_logger(path):
    logger = logging.getLogger("exp")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(path)
    logger.addHandler(fh)

    return logger