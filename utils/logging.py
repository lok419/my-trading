import logging

class CustomFormatter(logging.Formatter):
    green = "\x1b[32;20m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"    
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger(name='', log_dir=None):

    logger = logging.getLogger(name)
    logger.name = name
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())    

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.addHandler(ch)
    logger.propagate = False

    return logger

'''
    DO NOT USE 'logger'

    We need to export a default logger here. If there is some package (e.g. futu) which overrides the logger, we need to import this 'logger' after they import to override their logger
    Otherwise, we should always use 'get_logger()' to get the logger for your own application 
    This is an examples:

        from futu import *
        from utils.logging import logger, get_logger

'''

#logger = get_logger()