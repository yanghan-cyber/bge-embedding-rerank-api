import logging.config
import os

import torch
from colorlog import ColoredFormatter

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "colored": {
            "()": "colorlog.ColoredFormatter",
            "format": "%(asctime)s [%(log_color)s%(levelname)s%(reset)s]: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        },
        "default": {
            "format": "%(asctime)s [%(levelname)s]: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "colored"},
        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": "logs/app.log",
            "when": "d",
            "interval": 1,
            "backupCount": 14,
            "formatter": "default",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
        }
    },
}

# 创建日志目录（如果不存在）
os.makedirs(
    os.path.dirname(logging_config["handlers"]["file"]["filename"]), exist_ok=True
)

# 配置日志
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

# 其他全局配置
MODEL_TIMEOUT = 30  # 30分钟

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACCESS_TOKEN = None