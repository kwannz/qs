# -*- coding: utf-8 -*-
"""
File: dev_init.py
Author: peiqi
Date: 2025/5/14
Description: 
"""
from configparser import ConfigParser
import os
from panda_backtest import project_dir
from panda_backtest.system.panda_log import SRLogger
import panda_backtest
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
from common.config.config import config
import logging
from common.logging.system_logger import setup_logging

setup_logging()

class DevInit(object):
    def __init__(self):
        pass

    @classmethod
    def init_log_env(cls, file):
        # path = ProjectConfig.get_config_parser(project_dir).get('log', 'path')
        path = config["LOG_PATH"]
        # logger = logging.getLogger(__name__)
        # logger.info(f"日志环境初始化: file={file}, path={path}")

    @classmethod
    def init_remote_sr_log(cls, mock_id, opz_params_str, strategy_context):
        SRLogger.init_strategy_context(mock_id, opz_params_str, strategy_context)
        RemoteLogFactory.init_sr_logger(SRLogger)
        panda_backtest.SRLogger = RemoteLogFactory.get_sr_logger()

class ProjectConfig:
    _project_config_dict = dict()

    @staticmethod
    def get_config_parser(root_path):
        if root_path in ProjectConfig._project_config_dict.keys():
            return ProjectConfig._project_config_dict[root_path]
        else:
            # 获取全局
            all_config_filename = 'config.ini'
            all_file_path = os.path.join(root_path, all_config_filename)
            all_config_parser = ConfigParser()
            all_config_parser.read(all_file_path)
            config_file = all_config_parser.get('config', 'config_file')
            config_filename = ("config_%s.ini" % config_file)
            file_path = os.path.join(root_path, config_filename)
            _config_parser = ConfigParser()
            _config_parser.read(file_path, encoding='utf-8')
            ProjectConfig._project_config_dict[root_path] = _config_parser
            return _config_parser