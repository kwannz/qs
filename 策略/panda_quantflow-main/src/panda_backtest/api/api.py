from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.constant.strategy_constant import *
import pandas
from panda_backtest.backtest_common.type.order_type import *
from panda_backtest.backtest_common.risk.risk_api import RiskApi
from panda_backtest.system.panda_log import SRLogger
from panda_backtest.util.log.log_factory import LogFactory
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
api_list = []

logger = LogFactory.get_logger()
SRLogger = RemoteLogFactory.get_sr_logger()
risk_control_api = None
is_risk_control = False

def init_risk_control_api(api: RiskApi):
    global risk_control_api
    risk_control_api = api

def init_risk_control():
    global is_risk_control
    is_risk_control = True

def init_sr_logger():
    global SRLogger
    SRLogger = RemoteLogFactory.get_sr_logger()

def append_to_api_list(func):
    api_list.append(func.__name__)
    globals()[func.__name__] = func
    return func

@append_to_api_list
def order_shares(account_id, id_or_ins, quantity, style=MarketOrderStyle, retry_num=0, risk_control_client=None, remark=None):
    """
    指定股数交易
    :param retry_num:
    :param risk_control_client: 风控客户端
    :param account_id: 账号
    :param id_or_ins:  合约
    :param quantity:     需要落单的股数。正数代表买入，负数代表卖出。将会根据一手xx股来向下调整到一手的倍数，比如中国A股就是调整成100股的倍数
    :param style:      订单类型，默认是市价单
    :return:
    """
    context = CoreContext.get_instance()
    # if not context.strategy_context.is_stock_trade():
    #     SRLogger.error('当前不是股票交易时间')
    #     print('当前不是股票交易时间')
    #     return
    #
    # if not context.strategy_context.is_stock_trade():
    #     SRLogger.error('当前不是股票交易时间')
    #     print('当前不是股票交易时间')
    #     return

    # 处理用户输入的参数
    if quantity > 0:
        side = SIDE_BUY
        effect = OPEN
        quantity = quantity
    else:
        quantity = abs(quantity)
        side = SIDE_SELL
        effect = CLOSE

    idandins = id_or_ins.split('.')
    if len(idandins) != 2:
        # print('合约输入有误')
        SRLogger.error(str(id_or_ins) + '下单失败：合约输入有误')
        return

    ticker = idandins[0]
    market_name = idandins[1]

    if market_name == 'SH':
        market = MKT_SH
    elif market_name == 'SZ':
        market = MKT_SZ
    else:
        market = MKT_UNKNOWN

    price = 0
    price_type = MARKET
    if isinstance(style, LimitOrderStyle):
        price = style.limit_price
        price_type = LIMIT

    order = dict()
    order['symbol'] = id_or_ins
    order['ticker'] = ticker
    order['market'] = market
    order['price'] = price
    order['quantity'] = quantity
    order['price_type'] = price_type
    order['side'] = side
    order['effect'] = effect
    order['retry_num'] = retry_num
    order['now_system_order'] = 1
    order['order_insert_type'] = 0

    if risk_control_client is not None:
        order['now_system_order'] = 2
        order['risk_control_id'] = risk_control_client

    if remark is not None:
        order['remark'] = remark

    return context.operation_proxy.place_order(account_id, order)

@append_to_api_list
def order_values(account_id, id_or_ins, amount, style=MarketOrderStyle, retry_num=0, risk_control_client=None, remark=None):
    context = CoreContext.get_instance()
    if not context.strategy_context.is_stock_trade():
        SRLogger.error('当前不是股票交易时间')
        return []

    if not context.strategy_context.is_stock_trade():
        SRLogger.error('当前不是股票交易时间')
        return []

    # 处理用户输入的参数
    if amount > 0:
        side = SIDE_BUY
        effect = OPEN
        amount = amount
    else:
        amount = abs(amount)
        side = SIDE_SELL
        effect = CLOSE

    idandins = id_or_ins.split('.')
    if len(idandins) != 2:
        # print('合约输入有误')
        SRLogger.error(str(id_or_ins) + '下单失败：合约输入有误')
        return

    ticker = idandins[0]
    market_name = idandins[1]

    if market_name == 'SH':
        market = MKT_SH
    elif market_name == 'SZ':
        market = MKT_SZ
    else:
        market = MKT_UNKNOWN

    price = 0
    price_type = MARKET
    if isinstance(style, LimitOrderStyle):
        price = style.limit_price
        price_type = LIMIT

    order = dict()
    order['symbol'] = id_or_ins
    order['ticker'] = ticker
    order['market'] = market
    order['price'] = price
    order['amount'] = amount
    order['price_type'] = price_type
    order['side'] = side
    order['effect'] = effect
    order['retry_num'] = effect
    order['now_system_order'] = 1
    order['order_insert_type'] = 1

    if risk_control_client is not None:
        order['now_system_order'] = 2
        order['risk_control_id'] = risk_control_client

    if remark is not None:
        order['remark'] = remark

    return context.operation_proxy.place_order(account_id, order)

@append_to_api_list
def cancel_order(account_id, order_id):
    """
    撤单
    :param account_id:
    :param order_id:
    :return:
    """
    if order_id == '':
        SRLogger.error('订单号错误')
    context = CoreContext.get_instance()
    context.operation_proxy.cancel_order(account_id, order_id)

@append_to_api_list
def cancel_future_order(account_id, order_id):
    """
    撤单
    :param account_id:
    :param order_id:
    :return:
    """
    if order_id == '':
        SRLogger.error('订单号错误')
    context = CoreContext.get_instance()
    context.operation_proxy.cancel_future_order(account_id, order_id)

@append_to_api_list
def subscribe(symbol_list):
    context = CoreContext.get_instance()
    return context.operation_proxy.subscribe(symbol_list)

@append_to_api_list
def export_data_to_file(data_frame):
    if not isinstance(data_frame, pandas.DataFrame):
        SRLogger.error('保存数据文件错误：数据类型不为DataFrame')
        return
    context = CoreContext.get_instance()
    context.operation_proxy.export_data_to_file(data_frame)

@append_to_api_list
def buy_open(account_id, id_or_ins, amount, style=MarketOrderStyle, retry_num=None, risk_control_client=None,
             remark=None):
    context = CoreContext.get_instance()

    idandins = id_or_ins.split('.')
    if len(idandins) != 2:
        # print('合约输入有误')
        SRLogger.error(str(id_or_ins) + '下单失败：合约输入有误')
        return list()

    ticker = idandins[0]
    market_name = idandins[1]

    if market_name == 'CFFEX':
        market = 'CFFEX'
    elif market_name == 'CZCE':
        market = 'CZCE'
    elif market_name == 'DCE':
        market = 'DCE'
    elif market_name == 'SHFE':
        market = 'SHFE'
    elif market_name == 'INE':
        market = 'INE'
    elif market_name == 'GFEX':
        market = 'GFEX'
    else:
        SRLogger.error(str(id_or_ins) + '下单失败：交易所输入有误')
        return

    price = 0
    price_type = MARKET
    if isinstance(style, LimitOrderStyle):
        price = style.limit_price
        price_type = LIMIT

    order = dict()
    order['symbol'] = id_or_ins
    order['ticker'] = ticker
    order['market'] = market
    order['price'] = price
    order['quantity'] = amount
    order['price_type'] = price_type
    order['side'] = SIDE_BUY
    order['effect'] = OPEN
    order['now_system_order'] = 1
    order['is_td_close'] = 0
    if retry_num is not None:
        order['retry_num'] = retry_num
    else:
        order['retry_num'] = 0

    if risk_control_client is not None:
        order['now_system_order'] = 2
        order['risk_control_id'] = risk_control_client

    if remark is not None:
        order['remark'] = remark

    return context.operation_proxy.place_future_order(account_id, order)

@append_to_api_list
def sell_open(account_id, id_or_ins, amount, style=MarketOrderStyle, retry_num=None, risk_control_client=None,
              remark=None):
    context = CoreContext.get_instance()

    idandins = id_or_ins.split('.')
    if len(idandins) != 2:
        # print('合约输入有误')
        SRLogger.error(str(id_or_ins) + '下单失败：合约输入有误')
        return list()

    ticker = idandins[0]
    market_name = idandins[1]

    if market_name == 'CFFEX':
        market = 'CFFEX'
    elif market_name == 'CZCE':
        market = 'CZCE'
    elif market_name == 'DCE':
        market = 'DCE'
    elif market_name == 'SHFE':
        market = 'SHFE'
    elif market_name == 'INE':
        market = 'INE'
    elif market_name == 'GFEX':
        market = 'GFEX'
    else:
        SRLogger.error(str(id_or_ins) + '下单失败：交易所输入有误')
        return

    price = 0
    price_type = MARKET
    if isinstance(style, LimitOrderStyle):
        price = style.limit_price
        price_type = LIMIT

    order = dict()
    order['symbol'] = id_or_ins
    order['ticker'] = ticker
    order['market'] = market
    order['price'] = price
    order['quantity'] = amount
    order['price_type'] = price_type
    order['side'] = SIDE_SELL
    order['effect'] = OPEN
    order['now_system_order'] = 1
    order['is_td_close'] = 0
    if retry_num is not None:
        order['retry_num'] = retry_num
    else:
        order['retry_num'] = 0

    if risk_control_client is not None:
        order['now_system_order'] = 2
        order['risk_control_id'] = risk_control_client

    if remark is not None:
        order['remark'] = remark

    return context.operation_proxy.place_future_order(account_id, order)

@append_to_api_list
def sell_close(account_id, id_or_ins, amount, style=MarketOrderStyle, close_today=False, close_local=True,
               retry_num=None, risk_control_client=None, remark=None):
    context = CoreContext.get_instance()
    all_order_list = list()

    if not isinstance(amount, int):
        SRLogger.error('下单失败：下单手数应应为整数类型,输入值为' + str(amount))
        return all_order_list

    if amount <= 0:
        SRLogger.error(str(id_or_ins) + '平仓失败：平仓手数应大于0')
        return all_order_list

    idandins = id_or_ins.split('.')
    if len(idandins) != 2:
        # print('合约输入有误')
        SRLogger.error(str(id_or_ins) + '平仓失败：合约输入有误')
        return all_order_list

    ticker = idandins[0]
    market_name = idandins[1]

    if market_name == 'CFFEX':
        market = 'CFFEX'
    elif market_name == 'CZCE':
        market = 'CZCE'
    elif market_name == 'DCE':
        market = 'DCE'
    elif market_name == 'SHFE':
        market = 'SHFE'
    elif market_name == 'INE':
        market = 'INE'
    elif market_name == 'GFEX':
        market = 'GFEX'
    else:
        SRLogger.error(str(id_or_ins) + '下单失败：交易所输入有误')
        return

    price = 0
    price_type = MARKET
    if isinstance(style, LimitOrderStyle):
        price = style.limit_price
        price_type = LIMIT

    order = dict()
    order['symbol'] = id_or_ins
    order['ticker'] = ticker
    order['market'] = market
    order['price'] = price
    order['quantity'] = amount
    order['price_type'] = price_type
    order['side'] = SIDE_SELL
    order['effect'] = CLOSE
    order['now_system_order'] = 1
    if close_local:
        order['is_close_local'] = 1
    else:
        order['is_close_local'] = 0

    if close_today:
        order['is_td_close'] = 1
    else:
        order['is_td_close'] = 0

    if retry_num is not None:
        order['retry_num'] = retry_num
    else:
        order['retry_num'] = 0

    if risk_control_client is not None:
        order['now_system_order'] = 2
        order['risk_control_id'] = risk_control_client

    if remark is not None:
        order['remark'] = remark

    ctp_order_list = context.operation_proxy.place_future_order(account_id, order)
    all_order_list.extend(ctp_order_list)
    return all_order_list

@append_to_api_list
def buy_close(account_id, id_or_ins, amount, style=MarketOrderStyle, close_today=False, close_local=True,
              retry_num=None, risk_control_client=None, remark=None):
    all_order_list = list()
    context = CoreContext.get_instance()
    # if not context.strategy_context.is_future_trade():
    #     print('当前不是期货交易时间')
    #     SRLogger.error('当前不是期货交易时间')
    #     return all_order_list

    if not isinstance(amount, int):
        SRLogger.error('下单失败：下单手数应应为整数类型,输入值为' + str(amount))
        return all_order_list

    if amount <= 0:
        print(str(id_or_ins) + '平仓失败：平仓手数应大于0')
        SRLogger.error(str(id_or_ins) + '平仓失败：平仓手数应大于0')
        return all_order_list

    idandins = id_or_ins.split('.')
    if len(idandins) != 2:
        # print('合约输入有误')
        SRLogger.error(str(id_or_ins) + '平仓失败：合约输入有误')
        return all_order_list

    ticker = idandins[0]
    market_name = idandins[1]

    if market_name == 'CFFEX':
        market = 'CFFEX'
    elif market_name == 'CZCE':
        market = 'CZCE'
    elif market_name == 'DCE':
        market = 'DCE'
    elif market_name == 'SHFE':
        market = 'SHFE'
    elif market_name == 'INE':
        market = 'INE'
    elif market_name == 'GFEX':
        market = 'GFEX'
    else:
        SRLogger.error(str(id_or_ins) + '下单失败：交易所输入有误')
        return all_order_list

    price = 0
    price_type = MARKET
    if isinstance(style, LimitOrderStyle):
        price = style.limit_price
        price_type = LIMIT

    order = dict()
    order['symbol'] = id_or_ins
    order['ticker'] = ticker
    order['market'] = market
    order['price'] = price
    order['quantity'] = amount
    order['price_type'] = price_type
    order['side'] = SIDE_BUY
    order['effect'] = CLOSE
    order['now_system_order'] = 1
    if close_local:
        order['is_close_local'] = 1
    else:
        order['is_close_local'] = 0

    if close_today:
        order['is_td_close'] = 1
    else:
        order['is_td_close'] = 0

    if retry_num is not None:
        order['retry_num'] = retry_num
    else:
        order['retry_num'] = 0

    if risk_control_client is not None:
        order['now_system_order'] = 2
        order['risk_control_id'] = risk_control_client

    if remark is not None:
        order['remark'] = remark

    ctp_order_list = context.operation_proxy.place_future_order(account_id, order)
    if ctp_order_list is not None:
        all_order_list.extend(ctp_order_list)
    return all_order_list

@append_to_api_list
def cash_moving(from_account, to_account, cash, move_type):
    context = CoreContext.get_instance()
    return context.operation_proxy.cash_moving(from_account, to_account, cash, move_type)

@append_to_api_list
def get_today_order(account_id, order_id):
    context = CoreContext.get_instance()
    return context.operation_proxy.get_today_order(account_id, order_id)

@append_to_api_list
def get_today_future_order(account_id, order_id):
    context = CoreContext.get_instance()
    return context.operation_proxy.get_today_future_work_order(account_id, order_id)

@append_to_api_list
def get_today_work_order(account_id):
    context = CoreContext.get_instance()
    return context.operation_proxy.get_today_work_order(account_id)

@append_to_api_list
def get_today_work_future_order(account_id):
    context = CoreContext.get_instance()
    return context.operation_proxy.get_today_work_future_order(account_id)

@append_to_api_list
def long_future_target(account_id, id_or_ins, target_amount, style=MarketOrderStyle, risk_control_client=True,
                       remark=None):
    context = CoreContext.get_instance()
    if target_amount is None:
        return
    strategy_buy_quantity = context.strategy_context.future_account_dict[account_id].positions[
        id_or_ins].strategy_buy_quantity
    amount = target_amount - strategy_buy_quantity
    if amount == 0:
        return None
    if amount > 0:
        return buy_open(account_id, id_or_ins, amount, style=style, risk_control_client=risk_control_client,
                        remark=remark)
    else:
        return sell_close(account_id, id_or_ins, -amount, style=style, risk_control_client=risk_control_client,
                          remark=remark)

@append_to_api_list
def short_future_target(account_id, id_or_ins, target_amount, style=MarketOrderStyle, risk_control_client=True,
                        remark=None):
    context = CoreContext.get_instance()
    if target_amount is None:
        return
    strategy_sell_quantity = context.strategy_context.future_account_dict[account_id].positions[
        id_or_ins].strategy_sell_quantity
    amount = target_amount - strategy_sell_quantity
    if amount == 0:
        return None
    if amount > 0:
        return sell_open(account_id, id_or_ins, amount, style=style, risk_control_client=True, remark=remark)
    else:
        return buy_close(account_id, id_or_ins, -amount, style=style, risk_control_client=True, remark=remark)

@append_to_api_list
def auto_retry_cancel_future_order(order, max_retry=15):
    account = order.account

    SRLogger.error('订单被撤单，合约：%s, 撤单手数：%s，撤单信息：%s' % (str(order.order_book_id),
                                                     str(order.unfilled_quantity), str(order.message)))

    if hasattr(order, 'retry_num'):
        now_retry_num = order.retry_num + 1
        if now_retry_num > max_retry:
            SRLogger.error('订单追单失败，合约：%s, 追单手数：%s，原因：%s' % (str(order.order_book_id),
                                                            str(order.unfilled_quantity), '追单次数超过限制'))
            return
    else:
        now_retry_num = 1

    quantity = order.unfilled_quantity

    if order.effect == OPEN:
        if order.side == SIDE_BUY:
            order_list = buy_open(account, order.order_book_id, quantity,
                                  style=MarketOrderStyle, retry_num=now_retry_num)
        else:
            order_list = sell_open(account, order.order_book_id, quantity,
                                   style=MarketOrderStyle, retry_num=now_retry_num)
    else:
        if order.side == SIDE_BUY:

            if order.is_td_close == 0:
                order_list = buy_close(account, order.order_book_id, quantity,
                                       style=MarketOrderStyle, retry_num=now_retry_num)
            else:
                order_list = buy_close(account, order.order_book_id, quantity,
                                       style=MarketOrderStyle, close_today=True, retry_num=now_retry_num)

        else:

            if order.is_td_close == 0:
                order_list = sell_close(account, order.order_book_id, quantity,
                                        style=MarketOrderStyle, retry_num=now_retry_num)
            else:
                order_list = sell_close(account, order.order_book_id, quantity,
                                        style=MarketOrderStyle, close_today=True,
                                        retry_num=now_retry_num)

    return order_list

@append_to_api_list
def sub_stock(symbol_list):
    context = CoreContext.get_instance()
    context.operation_proxy.sub_stock(symbol_list)

@append_to_api_list
def purchase(account_id, symbol, amount, fund_cover_old=False, risk_control_client=None, remark=None):
    context = CoreContext.get_instance()
    order = dict()
    order['symbol'] = symbol
    order['side'] = SIDE_BUY
    order['effect'] = OPEN
    order['purchase_amount'] = amount
    if fund_cover_old:
        order['fund_cover_old'] = 1
    else:
        order['fund_cover_old'] = 0

    if risk_control_client is not None:
        order['now_system_order'] = 2
        order['risk_control_id'] = risk_control_client

    if remark is not None:
        order['remark'] = remark

    return context.operation_proxy.place_fund_order(account_id, order)

@append_to_api_list
def redeem(account_id, symbol, quantity, fund_cover_old=False, risk_control_client=None, remark=None):
    context = CoreContext.get_instance()
    order = dict()
    order['symbol'] = symbol
    order['side'] = SIDE_SELL
    order['effect'] = CLOSE
    order['quantity'] = quantity
    if fund_cover_old:
        order['fund_cover_old'] = 1
    else:
        order['fund_cover_old'] = 0

    if risk_control_client is not None:
        order['now_system_order'] = 2
        order['risk_control_id'] = risk_control_client

    if remark is not None:
        order['remark'] = remark

    return context.operation_proxy.place_fund_order(account_id, order)

@append_to_api_list
def draw(data):
    context = CoreContext.get_instance()
    return context.operation_proxy.draw(data)

@append_to_api_list
def target_future_group_order(account, long_symbol_dict, short_symbol_dict):
    context = CoreContext.get_instance()
    context.operation_proxy.insert_future_group_order(account, long_symbol_dict, short_symbol_dict)

@append_to_api_list
def target_stock_group_order(account, symbol_dict, price_type=0):
    context = CoreContext.get_instance()
    context.operation_proxy.insert_stock_group_order(account, symbol_dict, price_type)

@append_to_api_list
def pub_data(pub_key, json_data):
    if type(json_data) != str:
        print("策略推送的内容只能为字符串类型")
        return
    context = CoreContext.get_instance()
    context.operation_proxy.pub_data(pub_key, json_data)

@append_to_api_list
def sub_data(sub_keys, call_back):
    if type(sub_keys) != list:
        print("订阅key为list类型")
        return
    context = CoreContext.get_instance()
    context.operation_proxy.sub_data(sub_keys, call_back)