# *****************************************************************************
# Copyright (C) 2023 Ambra Di Piano
# This software is distributed under the terms of the BSD-3-Clause license
#
# Authors:
# Ambra Di Piano <ambra.dipiano@inaf.it>
# *****************************************************************************

import logging
from os import makedirs
from os.path import dirname, isfile, join

def set_logger(level, filename=None):
    log = logging.getLogger()
    log.setLevel(level)
    # console logger
    formatter = logging.Formatter('[%(asctime)s](%(levelname)s) %(name)s: %(message)s')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    log.addHandler(consoleHandler)
    # file logger
    if filename is not None:
        makedirs(dirname(filename), exist_ok=True)
        if not isfile(filename):
            f = open(filename, 'w+')
            f.close()
        fileHandler = logging.FileHandler(filename)
        fileHandler.setFormatter(formatter)
        log.addHandler(fileHandler)
    return log

def get_log_level(level):
    if level in ['DEBUG', 'debug', 10]:
        level = logging.DEBUG
    elif level in ['INFO', 'info', 20]:
        level = logging.INFO
    elif level in ['WARN', 'WARNING', 'warn', 'warning', 30]:
        level = logging.WARN
    elif level in ['ERROR', 'error', 40]:
        level = logging.ERROR
    elif level in ['CRITICAL', 'critical', 50]:
        level = logging.CRITICAL
    else:
        level = logging.NOTSET
    return level

def get_logfile(configuration, mode):
    logfile = configuration['logging']['logfile']
    if mode not in logfile:
        logfile = logfile.replace('.log', f'_{mode}.log')
    if configuration[mode]['output'] not in logfile:
        logfile = join(configuration[mode]['output'], logfile)
    return logfile