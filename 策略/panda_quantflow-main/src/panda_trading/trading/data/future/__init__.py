from panda_trading.trading.data.future.trade_time_update import update
from panda_trading.trading.data.future.trade_time_update import pz_dict
import logging
logger = logging.getLogger(__name__)

def initialize(config:pz_dict):
    # 初始化交易时间
    update(config)
    logger.info("交易时间段初始化完成")

print("initialize")
initialize(pz_dict)
