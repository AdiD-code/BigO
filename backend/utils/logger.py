import logging

def get_logger(name="backend"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Create formatter and add it to handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add handler to logger
    if not logger.handlers:  # Prevent duplicate handlers
        logger.addHandler(handler)

    return logger