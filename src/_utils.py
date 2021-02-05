import logging

from colorlog import ColoredFormatter

class ColorLogger:
    def __init__(self):
        """Return a logger with a ColoredFormatter."""
        self.formatter = ColoredFormatter(
            "%(log_color)s[%(asctime)s]%(levelname)s:%(name)s:%(funcName)s(%(lineno)s):%(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            reset=True,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            }
        )

        logger = logging.getLogger()
        self.root_logger = logger

        self.setStreamHandler()


    def setStreamHandler(self):
        handler = logging.StreamHandler()
        handler.setFormatter(self.formatter)
        self.root_logger.addHandler(handler)


    def setFileHandler(self, filename="log.log", log_level=10):
        handler = logging.FileHandler(filename=filename)
        handler.setFormatter(self.formatter)
        handler.setLevel(log_level)
        self.root_logger.addHandler(handler)


    def setLevel(self, log_level):
        self.root_logger.setLevel(log_level)
        return self.root_logger


    def __call__(self, name):
        return self.root_logger.getChild(name)


getLogger  = ColorLogger()
setLogLevel   = getLogger.setLevel
setLogFile = getLogger.setFileHandler
