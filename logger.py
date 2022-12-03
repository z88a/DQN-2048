import logging

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"

COLORS = {
    'DEBUG': BLUE,
    'INFO': WHITE,
    'WARNING': GREEN,
    'ERROR': RED,
    'CRITICAL': YELLOW
}

LEVELS = {
    1: logging.DEBUG,
    2: logging.INFO,
    3: logging.WARNING,
    4: logging.ERROR,
    5: logging.CRITICAL
}

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        message = str(record.msg)
        funcName = record.funcName
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            message_color = COLOR_SEQ % (30 + COLORS[levelname]) + message + RESET_SEQ
            funcName_color = COLOR_SEQ % (30 + COLORS[levelname]) + funcName + RESET_SEQ
            record.levelname = levelname_color
            record.msg = message_color
            record.funcName = funcName_color
        return logging.Formatter.format(self, record)


def logger(print_lev=1, save_lev=1, filename=None):
    # TODO 传参设置打印和存储level
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Log等级总开关

    LOGFORMAT = "[%(asctime)s][%(name)s] [%(levelname)s] (%(filename)s:%(funcName)s:%(lineno)d) %(message)s"
    # 创建一个handler，用于写入日志文件
    if filename is not None:
        logfile = './'+filename
        fh = logging.FileHandler(logfile, mode='a') # open的打开模式这里可以进行参考
        fh.setLevel(LEVELS[save_lev]) # 输出到file的log等级的开关
        fh.setFormatter(logging.Formatter(LOGFORMAT))
        logger.addHandler(fh)

    # 创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(LEVELS[print_lev])  # 输出到console的log等级的开关

    # 定义handler的输出格式
    formatter = ColoredFormatter(LOGFORMAT)
    ch.setFormatter(formatter)

    # 将logger添加到handler里面
    logger.addHandler(ch)
    return logger



def loghigher(func):
    def wrapper(*arg, **kw):
        logging.info(func.__name__+' start.')
        return func(*arg, **kw)
    return wrapper

'''
def loghigher():
    def decorator(func):
        def wrapper(*arg, **kw):
                logging.info(func.__name__+' start.')
                return func(*arg, **kw)
            return wrapper
        return decorator
'''