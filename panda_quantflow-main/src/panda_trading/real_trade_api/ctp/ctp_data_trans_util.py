import ctp as api
from ctp import  THOST_FTDC_OST_AllTraded, THOST_FTDC_OST_PartTradedQueueing, \
    THOST_FTDC_OST_PartTradedNotQueueing, THOST_FTDC_OST_NoTradeQueueing, THOST_FTDC_OST_NoTradeNotQueueing, \
    THOST_FTDC_OST_Canceled, THOST_FTDC_OST_Unknown, THOST_FTDC_OST_NotTouched, THOST_FTDC_OST_Touched, \
    THOST_FTDC_D_Buy, THOST_FTDC_OF_Open

from panda_backtest.backtest_common.constant.strategy_constant import *
from panda_backtest.backtest_common.model.result.order import Order
from panda_backtest.backtest_common.model.result.panda_backtest_instrument import PandaBacktestInstrument as XbBacktestInstrument
from panda_backtest.backtest_common.model.result.panda_backtest_trade import PandaBacktestTrade as XbBacktestTrade
from panda_backtest.backtest_common.model.result.panda_backtest_position import PandaBacktestPosition as XbBacktestPosition
from panda_backtest.backtest_common.model.result.panda_backtest_account import PandaBacktestAccount as XbBacktestAccount
from panda_backtest.backtest_common.model.result.panda_real_withdraw_deposit import PandaRealWithdrawDeposit as XbRealWithdrawDeposit
from panda_trading.real_trade_api.ctp.data.future_info_map import FutureInfoMap
from utils.data.symbol_util import SymbolUtil


class CTPDataTransUtil(object):
    def __init__(self):
        self.future_info_map = FutureInfoMap()
        pass

    def order_status_trans(self, pOrder):
        if pOrder.OrderStatus == THOST_FTDC_OST_AllTraded:
            pOrder.status = FILLED
        elif pOrder.OrderStatus == THOST_FTDC_OST_PartTradedQueueing:
            pOrder.status = PartTradedQueueing
        elif pOrder.OrderStatus == THOST_FTDC_OST_PartTradedNotQueueing:
            pOrder.status = PartTradedNotQueueing
        elif pOrder.OrderStatus == THOST_FTDC_OST_NoTradeQueueing:
            pOrder.status = NoTradeQueueing
        elif pOrder.OrderStatus == THOST_FTDC_OST_NoTradeNotQueueing:
            pOrder.status = NoTradeNotQueueing
        elif pOrder.OrderStatus == THOST_FTDC_OST_Canceled:
            pOrder.status = CANCELLED
        elif pOrder.OrderStatus == THOST_FTDC_OST_Unknown:
            pOrder.status = ACTIVE
        elif pOrder.OrderStatus == THOST_FTDC_OST_NotTouched:
            pOrder.status = NotTouched
        elif pOrder.OrderStatus == THOST_FTDC_OST_Touched:
            pOrder.status = Touched

    def order_data_trans(self, pOrder, order: Order):
        order.account = str(pOrder.InvestorID)
        order.order_client_id = str(pOrder.OrderRef).strip()
        order.order_id = str(pOrder.FrontID).strip() + str(pOrder.SessionID).strip() + str(pOrder.OrderRef).strip()
        if '&' in pOrder.InstrumentID:
            order.order_book_id = pOrder.InstrumentID
        elif pOrder.ExchangeID is None or pOrder.ExchangeID == '':
            instrument_info = self.future_info_map.get_by_ctp_code(pOrder.InstrumentID)
            order.order_book_id = instrument_info['symbol']
        else:
            order.order_book_id = SymbolUtil.ctp_code_to_code(pOrder.InstrumentID,
                                                              pOrder.ExchangeID)
        order.status = pOrder.status
        order.message = str(pOrder.StatusMsg)
        order.market = str(pOrder.ExchangeID).strip()
        order.order_sys_id = str(pOrder.OrderSysID).strip()
        order.filled_quantity = pOrder.VolumeTraded
        order.unfilled_quantity = pOrder.VolumeTotal
        order.relative_order_sys_iD = pOrder.RelativeOrderSysID
        order.front_id = pOrder.FrontID
        order.session_id = pOrder.SessionID
        order.price = pOrder.LimitPrice

        if pOrder.Direction == api.THOST_FTDC_D_Buy:
            order.side = SIDE_BUY
        else:
            order.side = SIDE_SELL

        if pOrder.CombOffsetFlag == api.THOST_FTDC_OF_Open:
            order.effect = OPEN
        else:
            if pOrder.CombOffsetFlag == api.THOST_FTDC_OF_CloseToday:
                order.effect = CLOSE
                order.is_td_close = 1
            else:
                order.effect = CLOSE
                order.is_td_close = 0

    def trade_data_trans(self, pTrade, trade: XbBacktestTrade):
        trade.trade_id = str(pTrade.TradeID).strip()
        trade.type = 1
        trade.account_id = str(pTrade.InvestorID)
        trade.order_sys_id = str(pTrade.OrderSysID).strip()
        # trade.order_id = str(pTrade.OrderRef).strip()
        trade.market = str(pTrade.ExchangeID).strip()
        # trade.contract_code = str(pTrade.InstrumentID).upper()
        # trade.contract_code = SymbolUtil.ctp_code_to_code(pTrade.InstrumentID, pTrade.ExchangeID)
        if pTrade.ExchangeID is None or pTrade.ExchangeID == '':
            instrument_info = self.future_info_map.get_by_ctp_code(pTrade.InstrumentID)
            trade.contract_code = instrument_info['symbol']
        else:
            trade.contract_code = SymbolUtil.ctp_code_to_code(pTrade.InstrumentID,
                                                              pTrade.ExchangeID)

        if pTrade.Direction == THOST_FTDC_D_Buy:
            trade.business = SIDE_BUY
        else:
            trade.business = SIDE_SELL

        trade.volume = pTrade.Volume
        trade.price = pTrade.Price

        if pTrade.OffsetFlag == THOST_FTDC_OF_Open:
            trade.direction = OPEN
        else:
            trade.direction = CLOSE
            if pTrade.OffsetFlag == api.THOST_FTDC_OF_CloseToday:
                trade.is_td_close = 1

        trade.gmt_create_time = str(pTrade.TradeTime)
        trade.gmt_create = str(pTrade.TradeDate)
        trade.run_type = 2

    def withdraw_deposit_trans(self, pRspTransfer, xb_real_withdraw_deposit:XbRealWithdrawDeposit):
        xb_real_withdraw_deposit.money = pRspTransfer.TradeAmount
        xb_real_withdraw_deposit.account_type = 1

    def position_data_trans(self, position_data, xb_back_test_position: XbBacktestPosition):

        if '&' in position_data.InstrumentID:
            # 组合合约
            instrument1 = position_data.InstrumentID.split('&')[1]
            instrument_info = self.future_info_map.get_by_ctp_code(instrument1)
            market = instrument_info['symbol'].split('.')[1]
            xb_back_test_position.contract_code = position_data.InstrumentID
            round_lot = 1
        else:
            if position_data.ExchangeID is None or position_data.ExchangeID == '':
                instrument_info = self.future_info_map.get_by_ctp_code(position_data.InstrumentID)
                xb_back_test_position.contract_code = instrument_info['symbol']
            else:
                xb_back_test_position.contract_code = SymbolUtil.ctp_code_to_code(position_data.InstrumentID,
                                                                                  position_data.ExchangeID)
                instrument_info = self.future_info_map[xb_back_test_position.contract_code]
            xb_back_test_position.type = 1
            if instrument_info is None:
                round_lot = 1
            else:
                round_lot = instrument_info['contractmul']
            xb_back_test_position.round_lot = round_lot

            market = xb_back_test_position.contract_code.split('.')[1]

        if market == 'SHF' or market == 'INE':
            # 上期所和能源交易所分昨仓和今仓返回
            xb_back_test_position.position += position_data.Position
            if position_data.YdPosition > 0:
                # 处理昨仓相关
                xb_back_test_position.yd_position += position_data.Position
            else:
                xb_back_test_position.yd_position += 0
            if position_data.TodayPosition > 0:
                # 处理今仓相关
                xb_back_test_position.td_position = position_data.TodayPosition
            else:
                xb_back_test_position.td_position += 0

            if str(position_data.PosiDirection) == '2':
                xb_back_test_position.direction = SIDE_BUY
                xb_back_test_position.sellable += position_data.Position - position_data.ShortFrozen
                xb_back_test_position.frozen_position += position_data.ShortFrozen
                if position_data.YdPosition > 0:
                    # 处理昨仓相关
                    xb_back_test_position.frozen_yd_position = position_data.ShortFrozen
                else:
                    xb_back_test_position.frozen_yd_position += 0
                if position_data.TodayPosition > 0:
                    # 处理今仓相关
                    xb_back_test_position.frozen_td_position = position_data.ShortFrozen
                else:
                    xb_back_test_position.frozen_td_position += 0

            elif str(position_data.PosiDirection) == '3':
                # 空头
                xb_back_test_position.direction = SIDE_SELL
                xb_back_test_position.sellable += position_data.Position - position_data.LongFrozen
                xb_back_test_position.frozen_position += position_data.LongFrozen
                if position_data.YdPosition > 0:
                    # 处理昨仓相关
                    xb_back_test_position.frozen_yd_position = position_data.LongFrozen
                else:
                    xb_back_test_position.frozen_yd_position += 0

                if position_data.TodayPosition > 0:
                    # 处理今仓相关
                    xb_back_test_position.frozen_td_position = position_data.LongFrozen
                else:
                    xb_back_test_position.frozen_td_position += 0
            else:
                xb_back_test_position.direction = 2

        else:
            xb_back_test_position.position += position_data.Position
            xb_back_test_position.td_position += position_data.TodayPosition
            xb_back_test_position.yd_position = position_data.Position - position_data.TodayPosition
            if str(position_data.PosiDirection) == '2':
                xb_back_test_position.direction = SIDE_BUY
                xb_back_test_position.sellable = position_data.Position - position_data.ShortFrozen
                xb_back_test_position.frozen_position = position_data.ShortFrozen

                if xb_back_test_position.yd_position >= position_data.ShortFrozen:
                    xb_back_test_position.frozen_yd_position = position_data.ShortFrozen
                else:
                    xb_back_test_position.frozen_yd_position = xb_back_test_position.yd_position
                    xb_back_test_position.frozen_td_position = position_data.ShortFrozen - xb_back_test_position.yd_position
            elif str(position_data.PosiDirection) == '3':
                # 空头
                xb_back_test_position.direction = SIDE_SELL
                xb_back_test_position.sellable = position_data.Position - position_data.LongFrozen
                xb_back_test_position.frozen_position = position_data.LongFrozen

                if xb_back_test_position.yd_position >= position_data.LongFrozen:
                    xb_back_test_position.frozen_yd_position = position_data.LongFrozen
                else:
                    xb_back_test_position.frozen_yd_position = xb_back_test_position.yd_position
                    xb_back_test_position.frozen_td_position = position_data.LongFrozen - xb_back_test_position.yd_position
            else:
                xb_back_test_position.direction = 2

        xb_back_test_position.open_cost += position_data.OpenCost

        xb_back_test_position.position_cost += position_data.PositionCost
        if xb_back_test_position.position == 0:
            xb_back_test_position.price = 0
            xb_back_test_position.hold_price = 0
        else:
            xb_back_test_position.price = xb_back_test_position.open_cost / (xb_back_test_position.position
                                                                                      * round_lot)
            xb_back_test_position.hold_price = xb_back_test_position.position_cost / (xb_back_test_position.position
                                                                                      * round_lot)
        xb_back_test_position.cost += position_data.Commission
        xb_back_test_position.margin += position_data.UseMargin
        xb_back_test_position.holding_pnl += position_data.PositionProfit
        xb_back_test_position.realized_pnl += position_data.CloseProfit

    def position_combine_data_trans(self, position_combine_data, xb_back_test_position: XbBacktestPosition):
        pass
        xb_back_test_position.account_id = position_combine_data.InvestorID
        xb_back_test_position.contract_code = position_combine_data.CombInstrumentID
        if position_combine_data.InstrumentID == position_combine_data.CombInstrumentID.split(' ')[1].split('&')[0]:
            if position_combine_data.Direction == '0':
                xb_back_test_position.direction = SIDE_BUY
            else:
                xb_back_test_position.direction = SIDE_SELL
        if position_combine_data.Margin > 0:
            xb_back_test_position.margin += position_combine_data.Margin
            xb_back_test_position.position += position_combine_data.TotalAmt
            if position_combine_data.Direction == '0':
                xb_back_test_position.buy_margin += position_combine_data.Margin
            else:
                xb_back_test_position.sell_margin += position_combine_data.Margin

    def asset_data_trans(self, asset_data, xb_back_test_account: XbBacktestAccount):
        xb_back_test_account.account_id = asset_data.AccountID
        xb_back_test_account.total_profit = asset_data.Balance
        xb_back_test_account.static_profit = asset_data.PreBalance
        xb_back_test_account.available_funds = asset_data.Available
        xb_back_test_account.margin = asset_data.CurrMargin
        xb_back_test_account.cost = asset_data.Commission
        xb_back_test_account.frozen_capital = asset_data.FrozenCash + asset_data.FrozenCommission + \
                                              asset_data.FrozenMargin
        # xb_back_test_account.holding_pnl = asset_data.PositionProfit
        xb_back_test_account.realized_pnl = asset_data.CloseProfit
        xb_back_test_account.today_deposit = asset_data.Deposit
        xb_back_test_account.today_withdraw = asset_data.Withdraw
        xb_back_test_account.daily_pnl = xb_back_test_account.holding_pnl + xb_back_test_account.realized_pnl - \
                                         xb_back_test_account.cost
        # xb_back_test_account.total_profit = xb_back_test_account.available_funds + \
        #     xb_back_test_account.frozen_capital + \
        #     xb_back_test_account.margin + asset_data.PositionProfit

    def instrument_data_trans(self, instrument_data, xb_back_test_instrument: XbBacktestInstrument):
        xb_back_test_instrument.type = 1
        xb_back_test_instrument.symbol = instrument_data.InstrumentID
        xb_back_test_instrument.market = instrument_data.ExchangeID
        xb_back_test_instrument.symbol = SymbolUtil.ctp_code_to_code(xb_back_test_instrument.symbol,
                                                                     xb_back_test_instrument.market)
        xb_back_test_instrument.name = instrument_data.InstrumentName
        xb_back_test_instrument.contractmul = instrument_data.VolumeMultiple
        xb_back_test_instrument.min_price_unit = instrument_data.PriceTick

    def instrument_margin_data_trans(self, instrument_margin_data, xb_back_test_instrument: XbBacktestInstrument):
        pass

    def input_order_data_trans(self, pInputOrder, order: Order):
        order.account = str(pInputOrder.InvestorID)
        order.order_client_id = str(pInputOrder.OrderRef).strip()
        # order.order_book_id = SymbolUtil.ctp_code_to_code(pInputOrder.InstrumentID, pInputOrder.ExchangeID)
        if pInputOrder.ExchangeID is None or pInputOrder.ExchangeID == '':
            instrument_info = self.future_info_map.get_by_ctp_code(pInputOrder.InstrumentID)
            order.order_book_id = instrument_info['symbol']
        else:
            order.order_book_id = SymbolUtil.ctp_code_to_code(pInputOrder.InstrumentID,
                                                              pInputOrder.ExchangeID)
        order.filled_quantity = 0
        order.unfilled_quantity = pInputOrder.VolumeTotalOriginal

        order.status = REJECTED

        if pInputOrder.Direction == api.THOST_FTDC_D_Buy:
            order.side = SIDE_BUY
        else:
            order.side = SIDE_SELL

        if pInputOrder.CombOffsetFlag == api.THOST_FTDC_OF_Open:
            order.effect = OPEN
        else:
            if pInputOrder.CombOffsetFlag == api.THOST_FTDC_OF_CloseToday:
                order.effect = CLOSE
                order.is_td_close = 1
            else:
                order.effect = CLOSE
                order.is_td_close = 0
