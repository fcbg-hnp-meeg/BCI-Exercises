'''
Initialize PyCNBI logger and other settings
TODO: support a process-safe file logging
Kyuhwa Lee
Swiss Federal Institute of Technology Lausanne (EPFL)
'''

import os
import sys
import logging
import pickle
import numpy as np


# log level options provided by pycnbi
LOG_LEVELS = {
    'DEBUG':logging.DEBUG,
    'INFO':logging.INFO,
    'INFO_GREEN':22,
    'INFO_BLUE':24,
    'INFO_YELLOW':26,
    'WARNING':logging.WARNING,
    'ERROR':logging.ERROR
}

class PycnbiFormatter(logging.Formatter):
    fmt_debug = "[%(module)s:%(funcName)s:%(lineno)d] DEBUG: %(message)s (%(asctime)s)"
    fmt_info = "[%(module)s.%(funcName)s] %(message)s"
    fmt_warning = "[%(module)s.%(funcName)s] WARNING: %(message)s"
    fmt_error = "[%(module)s:%(funcName)s:%(lineno)d] ERROR: %(message)s"

    def __init__(self, fmt="%(levelno)s: %(message)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == LOG_LEVELS['DEBUG']:
            self._fmt = self.fmt_debug
        elif record.levelno == LOG_LEVELS['INFO']:
            self._fmt = self.fmt_info
        elif record.levelno == LOG_LEVELS['INFO_GREEN']:
            self._fmt = self.fmt_info
        elif record.levelno == LOG_LEVELS['INFO_BLUE']:
            self._fmt = self.fmt_info
        elif record.levelno == LOG_LEVELS['INFO_YELLOW']:
            self._fmt = self.fmt_info
        elif record.levelno == LOG_LEVELS['WARNING']:
            self._fmt = self.fmt_warning
        elif record.levelno >= LOG_LEVELS['ERROR']:
            self._fmt = self.fmt_error
        self._style = logging.PercentStyle(self._fmt)
        return logging.Formatter.format(self, record)

def init_logger(verbose_console='INFO'):
    '''
    Add the first logger as sys.stdout. Handler will be added only once.
    '''
    if not logger.hasHandlers():
        add_logger_handler(sys.stdout, verbosity=verbose_console)
    
    '''
    TODO: add file handler
    # file logger handler
    f_handler = logging.FileHandler('pycnbi.log', mode='a')
    f_handler.setLevel(loglevels[verbose_file])
    f_format = logging.Formatter('%(levelname)s %(asctime)s %(funcName)s:%(lineno)d: %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    '''

def add_logger_handler(stream, verbosity='INFO'):
    # add custom log levels
    logging.addLevelName(LOG_LEVELS['INFO_GREEN'], 'INFO_GREEN')
    def __log_info_green(self, message, *args, **kwargs):
        if self.isEnabledFor(LOG_LEVELS['INFO_GREEN']):
            self._log(LOG_LEVELS['INFO_GREEN'], message, args, **kwargs)
    logging.Logger.info_green = __log_info_green

    logging.addLevelName(LOG_LEVELS['INFO_BLUE'], 'INFO_BLUE')
    def __log_info_blue(self, message, *args, **kwargs):
        if self.isEnabledFor(LOG_LEVELS['INFO_BLUE']):
            self._log(LOG_LEVELS['INFO_BLUE'], message, args, **kwargs)
    logging.Logger.info_blue = __log_info_blue

    logging.addLevelName(LOG_LEVELS['INFO_YELLOW'], 'INFO_YELLOW')
    def __log_info_yellow(self, message, *args, **kwargs):
        if self.isEnabledFor(LOG_LEVELS['INFO_YELLOW']):
            self._log(LOG_LEVELS['INFO_YELLOW'], message, args, **kwargs)
    logging.Logger.info_yellow = __log_info_yellow

    # console logger handler
    c_handler = logging.StreamHandler(stream)
    c_handler.setFormatter(PycnbiFormatter())
    logger.addHandler(c_handler)

    # minimum possible level of all handlers
    logger.setLevel(logging.DEBUG)

    logger.handlers[-1].level = LOG_LEVELS[verbosity]
    set_log_level(verbosity)
    return logger

def set_log_level(verbosity, handler_id=0):
    '''
    hander ID 0 is always stdout, followed by user-defined handlers.
    '''
    logger.handlers[handler_id].level = LOG_LEVELS[verbosity]

def parse_path(file_path):
    """
    Input:
        full path
    Returns:
        self.dir = base directory of the file
        self.name = file name without extension
        self.ext = file extension
    """
    class path_info:
        def __init__(self, path):
            path_abs = os.path.realpath(path).replace('\\', '/')
            s = path_abs.split('/')
            f = s[-1].split('.')
            basedir = '/'.join(s[:-1])
            if len(f) == 1:
                name, ext = f[-1], ''
            else:
                name, ext = '.'.join(f[:-1]), f[-1]
            self.dir = basedir
            self.name = name
            self.ext = ext
            self.txt = 'self.dir=%s\nself.name=%s\nself.ext=%s\n' % (self.dir, self.name, self.ext)
        def __repr__(self):
            return self.txt
        def __str__(self):
            return self.txt

    return path_info(file_path)

# set loggers
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logger = logging.getLogger('bci_exercises')
logger.propagate = False
init_logger()

#----------------------------------------------------------------------   
def feature2chz(x, fqlist, ch_names):
    """
    Label channel, frequency pair for PSD feature indices

    Input
    =====
    x: feature index
    fqlist: list of frequency bands
    ch_names: list of complete channel names

    Output
    ======
    (channel, frequency)

    """

    x = np.array(x).astype('int64').reshape(-1)
    fqlist = np.array(fqlist).astype('float64')
    ch_names = np.array(ch_names)

    n_fq = len(fqlist)
    hz = fqlist[x % n_fq]
    ch = (x / n_fq).astype('int64')  # 0-based indexing

    return ch_names[ch], hz

#----------------------------------------------------------------------
def save_obj(fname, obj, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Save python object into a file
    Set protocol=2 for Python 2 compatibility
    """
    with open(fname, 'wb') as fout:
        pickle.dump(obj, fout, protocol)

#----------------------------------------------------------------------
def sort_by_value(s, rev=False):
    """
    Sort dictionary or list by value and return a sorted list of keys and values.
    Values must be hashable and unique.
    """
    assert type(s) == dict or type(s) == list, 'Input must be a dictionary or list.'
    if type(s) == list:
        s = dict(enumerate(s))
    s_rev = dict((v, k) for k, v in s.items())
    values = sorted(s_rev, reverse=rev)
    keys = [s_rev[x] for x in values]
    return keys, values

#----------------------------------------------------------------------
def parse_path_list(path):
    """
    Input:
        full path
    Returns:
        base dir, file(or dir) name, extension (if file)
    """

    path_abs = os.path.realpath(path).replace('\\', '/')
    s = path_abs.split('/')
    f = s[-1].split('.')
    basedir = '/'.join(s[:-1]) + '/'
    if len(f) == 1:
        name, ext = f[-1], ''
    else:
        name, ext = '.'.join(f[:-1]), f[-1]

    return basedir, name, ext

#----------------------------------------------------------------------
def get_file_list(path, fullpath=True, recursive=False):
    """
    Get files with or without full path.
    """
    path = path.replace('\\', '/')
    if not path[-1] == '/': path += '/'

    if recursive == False:
        if fullpath == True:
            filelist = [path + f for f in os.listdir(path) if os.path.isfile(path + '/' + f) and f[0] != '.']
        else:
            filelist = [f for f in os.listdir(path) if os.path.isfile(path + '/' + f) and f[0] != '.']
    else:
        filelist = []
        for root, dirs, files in os.walk(path):
            root = root.replace('\\', '/')
            if fullpath == True:
                [filelist.append(root + '/' + f) for f in files]
            else:
                [filelist.append(f) for f in files]
    return sorted(filelist)
