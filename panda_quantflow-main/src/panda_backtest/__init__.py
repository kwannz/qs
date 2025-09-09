# -*- coding: utf-8 -*-
"""
File: __init__.py
Author: peiqi
Date: 2025/5/14
Description: 
"""

import os
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory

project_dir = os.path.dirname(os.path.abspath(__file__))

SRLogger = RemoteLogFactory.get_sr_logger()
