#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-2-2 下午2:43
# @Author : wlb
# @File   : xtp_order_data.py
# @desc   : 订单实体类
import dataclasses
import json
import logging

from panda_backtest.backtest_common.constant.strategy_constant import *
@dataclasses.dataclass
class Order:

    def __init__(self):
        self.order_id = EMPTY_STRING                # 唯一标识订单的id
        self.account = EMPTY_STRING                 # 账号
        self.order_book_id = EMPTY_STRING           # 合约代码
        self.order_client_id = EMPTY_STRING         # 合约客户端代码
        self.client_id = EMPTY_STRING               # 标识具体来源（在实盘策略中标识策略id，在风控界面手动下单上标识具体风险订单object_id）
        self.order_book_name = EMPTY_STRING         # 合约名称
        self.datetime = EMPTY_STRING                # 订单创建时间
        self.date = EMPTY_INT                       # 订单创建时间
        self.trade_date = EMPTY_INT                 # 订单交易日
        self.side = None                            # 订单方向（1：买 2：卖）
        self.price = EMPTY_FLOAT                    # 订单价格，只有在订单类型为'限价单'的时候才有意义
        self.quantity = EMPTY_INT                   # 订单数量
        self.close_td_pos = EMPTY_INT               # 平今仓位
        self.filled_quantity = EMPTY_INT            # 订单已成交数量
        self.unfilled_quantity = EMPTY_INT          # 订单未成交数量
        self.filled_close_td_pos = EMPTY_INT        # 订单未成交平今仓数量
        self.unfilled_close_td_pos = EMPTY_INT      # 订单未成交平今仓数量
        self.price_type = None
        self.transaction_cost = EMPTY_FLOAT         # 费用
        self.avg_price = EMPTY_FLOAT                # 成交均价
        self.status = None                          # 订单状态 （3：撤单）
        self.message = EMPTY_STRING                 # 信息。比如拒单时候此处会提示拒单原因
        self.trading_datetime = EMPTY_STRING        # 订单的交易日期（对应期货夜盘）
        self.effect = EMPTY_INT                     # 订单开平（期货专用， 0：开，1：平）
        self.margin = EMPTY_FLOAT
        self.is_td_close = EMPTY_INT                # 是否今日平仓(1：平今)
        self.order_sys_id = EMPTY_STRING            # 期货CTP使用(用于撤单)
        self.market = EMPTY_STRING                  # 市场（实盘用到）
        self.now_system_order = EMPTY_INT           # 是否为当前系统进程所下单子（0：外部订单，1：策略下单，2：风控下单, 3:平台手动下单）
        self.order_type = EMPTY_INT                 # 订单类型 （0:股票，1：期货, 2: 基金）
        self.retry_num = EMPTY_INT                  # 追单次数
        self.is_close_local = EMPTY_INT             # 是否为平本地仓位,(0:平本地仓位，1：平真实仓位)
        self.fund_cover_old = EMPTY_INT             # 是否覆盖老基金订单（0：否，1：是）
        self.latency_date = EMPTY_INT               # 基金赎回到账天数
        self.fund_type = EMPTY_INT                  # 覆盖老基金订单
        self.purchase_amount = EMPTY_INT
        self.fund_cross_date = EMPTY_INT            # 基金撮合日期
        self.fund_arrive_date = EMPTY_STRING        # 基金到账日
        self.redeem_frozen_amount = EMPTY_FLOAT     # 赎回冻结资金
        self.cur_filled_quantity = EMPTY_INT        # 部分成交本次成交数量
        self.cur_close_td_pos = EMPTY_INT           # 部分成交本次平今仓位
        self.risk_control_id = EMPTY_STRING         # 风控id
        self.remark = EMPTY_INT                     # 订单用户自定义备注

        self.relative_order_sys_iD = EMPTY_STRING   # 关联订单
        self.front_id = EMPTY_STRING
        self.session_id = EMPTY_STRING
        self.run_id = EMPTY_STRING
        self.is_reject_by_risk = EMPTY_INT          # 是否风控拒单，0：否，1：是
        self.stock_type = EMPTY_INT

if __name__ == '__main__':
    order=Order()
    print("",json.dumps(order.__dict__))
