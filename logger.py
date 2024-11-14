import logging
import os
from colorama import Fore, Style
from datetime import datetime
from colorama import init

init(autoreset=True)

MAX_LOG = 50
LOG_DIR = "logs"

class ColorFormatter(logging.Formatter):
    def format(self, record):
        record.pathname = os.path.relpath(record.pathname)
        log_msg = super().format(record)

        if record.levelname == "INFO":
            log_msg = log_msg.replace("INFO", f"{Fore.GREEN}INFO{Style.RESET_ALL}")
        elif record.levelname == "DEBUG":
            log_msg = log_msg.replace("DEBUG", f"{Fore.BLUE}DEBUG{Style.RESET_ALL}")
        elif record.levelname == "WARNING":
            log_msg = log_msg.replace("WARNING", f"{Fore.YELLOW}WARNING{Style.RESET_ALL}")
        elif record.levelname == "ERROR":
            log_msg = log_msg.replace("ERROR", f"{Fore.RED}ERROR{Style.RESET_ALL}")
        elif record.levelname == "CRITICAL":
            log_msg = log_msg.replace("CRITICAL", f"{Fore.MAGENTA}CRITICAL{Style.RESET_ALL}")

        return log_msg

def manage_log_files(log_dir, max_log):
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]

    log_files = sorted(
        log_files,
        key=lambda f: datetime.strptime(f.split(".")[0], "%Y-%m-%d")
    )

    while len(log_files) > max_log:
        try:
            oldest_file = log_files.pop(0)
            os.remove(os.path.join(log_dir, oldest_file))
        except Exception:
            pass

def set_log_level(logger, level):
    if not hasattr(logger, 'console_handler'):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = ColorFormatter(fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s", datefmt="%H:%M:%S")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.console_handler = console_handler  # console_handler 속성 설정

    logger.console_handler.setLevel(level)

def get_logger(console_level=logging.INFO, max_log=MAX_LOG):
    logger = logging.getLogger("logger")

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    formatter = ColorFormatter(fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)

    os.makedirs(LOG_DIR, exist_ok=True)

    log_filename = datetime.now().strftime("%Y-%m-%d.log")
    file_path = os.path.join(LOG_DIR, log_filename)

    file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s", datefmt="%H:%M:%S")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    manage_log_files(LOG_DIR, max_log)

    # console_handler 속성 설정
    logger.console_handler = console_handler

    return logger
