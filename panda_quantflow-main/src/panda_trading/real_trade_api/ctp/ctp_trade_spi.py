import ctp as api
import time
from datetime import datetime
import traceback

from panda_backtest.backtest_common.model.result.order import Order
from panda_backtest.backtest_common.constant.strategy_constant import *

from panda_trading.real_trade_api.ctp.ctp_data_trans_util import CTPDataTransUtil

from panda_backtest.backtest_common.model.result.panda_backtest_instrument import PandaBacktestInstrument as XbBacktestInstrument
from panda_backtest.backtest_common.model.result.panda_backtest_trade import PandaBacktestTrade as XbBacktestTrade
from panda_backtest.backtest_common.model.result.panda_backtest_position import PandaBacktestPosition as XbBacktestPosition
from panda_backtest.backtest_common.model.result.panda_backtest_account import PandaBacktestAccount as XbBacktestAccount
from panda_backtest.backtest_common.model.result.panda_real_withdraw_deposit import PandaRealWithdrawDeposit as XbRealWithdrawDeposit

from utils.data.symbol_util import SymbolUtil
from utils.log.log_factory import LogFactory


class CTPTraderSpi(api.CThostFtdcTraderSpi):

    def __init__(self, ctp_trade_api, account, is_auth_flag):
        api.CThostFtdcTraderSpi.__init__(self)
        self.ctp_trade_api = ctp_trade_api
        self.account = account
        self.trade_adapter = ctp_trade_api.trade_adapter
        self.ls_position_data_dict = dict()
        self.ls_position_combine_data_dict = dict()
        self.ctp_data_trans_util = CTPDataTransUtil()
        self.is_auth_flag = is_auth_flag
        self.position_req_id = None
        self.logger = LogFactory.get_logger()

    def OnFrontConnected(self):
        """当客户端与交易后台建立起通信连接时（还未登录前），该方法被调用。"""
        self.logger.info('ctp柜台连接')
        try:
            if self.is_auth_flag == 1:
                self.ctp_trade_api.account_auth(str(self.account))
            else:
                self.ctp_trade_api.account_login(self.account)
        except Exception as e:
            mes = traceback.format_exc()
            self.logger.error('OnFrontConnected异常:', str(mes))

    def OnFrontDisconnected(self, nReason):
        """当客户端与交易后台通信连接断开时，该方法被调用。当发生这个情况后，API会自动重新连接，客户端可不做处理。
        @param nReason 错误原因
                0x1001 网络读失败
                0x1002 网络写失败
                0x2001 接收心跳超时
                0x2002 发送心跳失败
                0x2003 收到错误报文
        """
        try:
            self.logger.info('OnFrontDisconnected')
            self.logger.info('断开连接')
            if self.account in self.ctp_trade_api.ctp_qry_thread_dict.keys():
                self.ctp_trade_api.ctp_qry_thread_dict[self.account].change_con_status(self.account, False)
        except Exception as e:
            self.logger.error('OnFrontConnected:', str(e))

    def OnHeartBeatWarning(self, nTimeLapse):
        """心跳超时警告。当长时间未收到报文时，该方法被调用。
        @param nTimeLapse 距离上次接收报文的时间
        """
        self.logger.info('心跳')

    def OnRspAuthenticate(self, pRspAuthenticate, pRspInfo, nRequestID, bIsLast):
        """客户端认证响应"""
        self.logger.info('OnRspAuthenticate')
        if pRspInfo and pRspInfo.ErrorID != 0:
            self.logger.info('期货账号' + str(self.account) + '认证失败，原因：' + str(pRspInfo.ErrorMsg))
            return
        self.logger.info('期货账号' + str(self.account) + '成功认证')
        # self.ctp_trade_api.req_user_password_update(self.account)
        self.ctp_trade_api.account_login(self.account)

    def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):
        """登录请求响应"""
        try:
            self.logger.info('OnRspUserLogin')
            self.logger.info(pRspInfo.ErrorID)
            if pRspInfo and pRspInfo.ErrorID != 0:
                self.logger.info('期货账号' + str(self.account) + '登录失败，原因：' + str(pRspInfo.ErrorMsg))
                return
            self.logger.info('期货账号' + str(self.account) + '成功登录')
            self.trade_adapter.on_account_login(self.account)
            if pRspUserLogin is None:
                self.ctp_trade_api.init_ctp_order_id(int(100), self.account)
            else:
                self.ctp_trade_api.init_account_front_session(self.account, pRspUserLogin.FrontID,
                                                              pRspUserLogin.SessionID)
                if pRspUserLogin.MaxOrderRef == '':
                    self.ctp_trade_api.init_ctp_order_id(int(1), self.account)
                else:
                    self.ctp_trade_api.init_ctp_order_id(int(pRspUserLogin.MaxOrderRef), self.account)

            self.ctp_trade_api.account_settle_confirm(self.account)
        except Exception:
            self.logger.error('OnRspUserLogin，错误日志:%s' % str(traceback.format_exc()))

    def OnRspUserLogout(self, pUserLogout, pRspInfo, nRequestID, bIsLast):
        """登出请求响应"""
        try:
            self.logger.info('账号成功登出，账号：%s' % str(self.account))
        except Exception as e:
            self.logger.error('CTP账号成功登出，错误日志:%s', str(traceback.format_exc()))

    def OnRspQryTradingAccount(self, pTradingAccount, pRspInfo, nRequestID, bIsLast):
        # print('OnRspQryTradingAccount')
        self.logger.info('OnRspQryTradingAccount, Available=>' + str(pTradingAccount.Available))
        self.logger.info('OnRspQryTradingAccount, PreBalance=>' + str(pTradingAccount.PreBalance))
        self.logger.info('OnRspQryTradingAccount, Commission=>' + str(pTradingAccount.Commission))
        self.logger.info('OnRspQryTradingAccount, Withdraw=>' + str(pTradingAccount.Withdraw))
        self.logger.info('OnRspQryTradingAccount, Deposit=>' + str(pTradingAccount.Deposit))
        try:
            if pTradingAccount is None:
                return
            xb_back_test_account = XbBacktestAccount()
            self.ctp_data_trans_util.asset_data_trans(pTradingAccount, xb_back_test_account)
            self.trade_adapter.on_rsp_qry_trading_account(self.account, xb_back_test_account)
            if self.account in self.ctp_trade_api.ctp_qry_thread_dict.keys():
                self.ctp_trade_api.ctp_qry_thread_dict[self.account].qry_end()
        except Exception as e:
            self.logger.error('查询账号资金失败，错误日志:%s' % str(traceback.format_exc()))

    def OnRspQryInvestorPosition(self, pInvestorPosition, pRspInfo, nRequestID, bIsLast):
        try:
            self.logger.info('OnRspQryInvestorPosition')
            account_id = self.account
            if (pInvestorPosition is None or pInvestorPosition.InstrumentID == '') and bIsLast:
                self.trade_adapter.on_rsp_qry_investor_position(self.account, {}, nRequestID)
                if account_id in self.ls_position_data_dict.keys():
                    self.ls_position_data_dict.pop(account_id)
                self.position_req_id = None
                if self.account in self.ctp_trade_api.ctp_qry_thread_dict.keys():
                    self.ctp_trade_api.ctp_qry_thread_dict[self.account].qry_end()
                return
            # self.logger.info('=======================持仓回报==========================')
            # self.logger.info('合约：' + str(pInvestorPosition.InstrumentID))
            # self.logger.info('仓位：', pInvestorPosition.Position)
            # print('昨仓：', pInvestorPosition.YdPosition)
            # print('今仓：', pInvestorPosition.TodayPosition)
            # print('平多仓：', pInvestorPosition.LongFrozen)
            # print('平空仓：', pInvestorPosition.ShortFrozen)
            # self.logger.info('方向：' + str(pInvestorPosition.PosiDirection))
            # self.logger.info('ExchangeMargin：' + str(pInvestorPosition.ExchangeMargin))
            # self.logger.info('PreMargin：' + str(pInvestorPosition.PreMargin))
            # self.logger.info('UseMargin：' + str(pInvestorPosition.UseMargin))
            # self.logger.info('FrozenMargin：' + str(pInvestorPosition.FrozenMargin))
            # self.logger.info('MarginRateByMoney：' + str(pInvestorPosition.MarginRateByMoney))
            # self.logger.info('MarginRateByVolume：' + str(pInvestorPosition.MarginRateByVolume))
            # print('请求id：', nRequestID)
            # print('结束标志位：', bIsLast)
            # self.logger.info('=======================持仓回报==========================')
            # 保存组合合约
            if '&' in pInvestorPosition.InstrumentID and self.account in self.ctp_trade_api.ctp_qry_thread_dict.keys():
                if pInvestorPosition.Position > 0:
                    self.ctp_trade_api.ctp_qry_thread_dict[self.account].save_zh_symbol(
                        pInvestorPosition.InstrumentID)
                else:
                    self.ctp_trade_api.ctp_qry_thread_dict[self.account].del_zh_symbol(
                        pInvestorPosition.InstrumentID)

            if account_id in self.ls_position_data_dict.keys():
                account_position_ls_dict = self.ls_position_data_dict[account_id]
                # 防止CTP不返回最后标志位的情况
                if self.position_req_id is not None and self.position_req_id != nRequestID:
                    # print('能够进行持仓回报了')
                    self.trade_adapter.on_rsp_qry_investor_position(self.account, account_position_ls_dict,
                                                                    self.position_req_id)
                    self.position_req_id = nRequestID
                    account_position_ls_dict = dict()
                else:
                    self.position_req_id = nRequestID
            else:
                account_position_ls_dict = dict()
                self.ls_position_data_dict[account_id] = account_position_ls_dict

            symbol = SymbolUtil.ctp_code_to_code(pInvestorPosition.InstrumentID,
                                                 pInvestorPosition.ExchangeID)
            key = symbol + '_' + pInvestorPosition.PosiDirection
            if key in account_position_ls_dict.keys():
                xb_back_test_position = account_position_ls_dict[key]
            else:
                xb_back_test_position = XbBacktestPosition()
                account_position_ls_dict[key] = xb_back_test_position
            self.ctp_data_trans_util.position_data_trans(pInvestorPosition, xb_back_test_position)
            if bIsLast:
                self.trade_adapter.on_rsp_qry_investor_position(self.account, account_position_ls_dict, nRequestID)
                self.ls_position_data_dict.pop(account_id)
                self.position_req_id = None
                if self.account in self.ctp_trade_api.ctp_qry_thread_dict.keys():
                    self.ctp_trade_api.ctp_qry_thread_dict[self.account].qry_end()
        except Exception as e:
            self.logger.error('查询持仓失败，错误日志:%s' % str(traceback.format_exc()))

    def OnRspQryInvestorPositionCombineDetail(self, pInvestorPositionCombineDetail, pRspInfo, nRequestID, bIsLast):
        try:
            if (pInvestorPositionCombineDetail is None or pInvestorPositionCombineDetail.InstrumentID == '') \
                    and bIsLast:
                self.ctp_trade_api.ctp_qry_thread_dict[self.account].qry_end()
                return
            self.logger.info('-------------------组合持仓回报-------------------')
            self.logger.info('bIsLast:{}'.format(bIsLast))
            self.logger.info('InstrumentID:{}'.format(pInvestorPositionCombineDetail.InstrumentID))
            self.logger.info('Direction:{}'.format(pInvestorPositionCombineDetail.Direction))
            self.logger.info('Margin:{}'.format(pInvestorPositionCombineDetail.Margin))
            self.logger.info('TotalAmt:{}'.format(pInvestorPositionCombineDetail.TotalAmt))
            self.logger.info('ComTradeID:{}'.format(pInvestorPositionCombineDetail.ComTradeID))
            self.logger.info('TradeID:{}'.format(pInvestorPositionCombineDetail.TradeID))
            self.logger.info('ExchangeID:{}'.format(pInvestorPositionCombineDetail.ExchangeID))
            self.logger.info('CombInstrumentID:{}'.format(pInvestorPositionCombineDetail.CombInstrumentID))
            self.logger.info('-------------------组合持仓回报-------------------')
            key = pInvestorPositionCombineDetail.CombInstrumentID
            account_id = self.account
            if account_id in self.ls_position_combine_data_dict.keys():
                account_position_ls_dict = self.ls_position_combine_data_dict[account_id]
            else:
                account_position_ls_dict = dict()
                self.ls_position_combine_data_dict[account_id] = account_position_ls_dict
            if key in account_position_ls_dict.keys():
                xb_back_test_position = account_position_ls_dict[key]
            else:
                xb_back_test_position = XbBacktestPosition()
                account_position_ls_dict[key] = xb_back_test_position

            self.ctp_data_trans_util.position_combine_data_trans(pInvestorPositionCombineDetail, xb_back_test_position)
            if xb_back_test_position.position > 0:
                self.ctp_trade_api.ctp_qry_thread_dict[self.account].save_zh_symbol(
                    xb_back_test_position.contract_code)
            else:
                self.ctp_trade_api.ctp_qry_thread_dict[self.account].del_zh_symbol(
                    xb_back_test_position.contract_code)
            if bIsLast:
                self.trade_adapter.on_rsp_qry_investor_position(self.account, account_position_ls_dict, 0)
                self.ls_position_combine_data_dict.pop(account_id)
                self.position_req_id = None
                if self.account in self.ctp_trade_api.ctp_qry_thread_dict.keys():
                    self.ctp_trade_api.ctp_qry_thread_dict[self.account].qry_end()
        except Exception as e:
            self.logger.error('查询组合持仓明细失败，错误日志:%s' % str(traceback.format_exc()))

    def OnRspSettlementInfoConfirm(self, pSettlementInfoConfirm, pRspInfo, nRequestID, bIsLast):
        self.logger.info('OnRspSettlementInfoConfirm')
        self.logger.info('期货账号' + str(self.account) + '成功结算')
        if self.account in self.ctp_trade_api.ctp_qry_thread_dict.keys():
            self.logger.info('更新状态')
            self.ctp_trade_api.ctp_qry_thread_dict[self.account].change_con_status(self.account, True)

    def OnRspQryExchangeMarginRate(self, pExchangeMarginRate, pRspInfo, nRequestID, bIsLast):
        self.logger.info(pExchangeMarginRate)

    def OnRspOrderInsert(self, pInputOrder, pRspInfo, nRequestID, bIsLast):
        self.logger.info('OnRspOrderInsert')
        self.logger.info('OnRspOrderInsert报单错误回报' + str(pRspInfo.ErrorMsg))
        try:
            # 是否为当前进程发出的报单标识
            if pInputOrder is None:
                return

            account_info = self.ctp_trade_api.get_ctp_account_info(self.account)
            if account_info is None:
                return
            order_id = str(account_info['front_id']) + str(account_info['session_id']) + str(pInputOrder.OrderRef)
            old_order = self.trade_adapter.get_work_order_by_order_id(self.account, order_id)

            if old_order is None:
                old_order = Order()
                old_order.client_id = '0000001'
                old_order.order_id = str('0000001') + str(pInputOrder.OrderRef)
                old_order.now_system_order = 0

            old_order.message = pRspInfo.ErrorMsg
            old_order.status = REJECTED

            self.ctp_data_trans_util.input_order_data_trans(pInputOrder, old_order)
            self.trade_adapter.on_rtn_order(self.account, old_order)

            self.trade_adapter.on_system_cancel_order(self.account, old_order)

            if pRspInfo is None:
                return

            if pRspInfo.ErrorID != 0:
                res_mes = 'OnRspOrderInsert错误,账号:' + str(self.account) + ',股票代码:' + pInputOrder.InstrumentID + \
                          ',错误码:' + str(pRspInfo.ErrorID) + '错误信息:' + pRspInfo.ErrorMsg
                self.logger.info(res_mes)
        except Exception as e:
            self.logger.error('插入报单异常，错误日志:%s' % str(traceback.format_exc()))

    def OnRtnOrder(self, pOrder):
        self.logger.info('OnRtnOrder')
        # time.sleep(10)
        # 订单状态变动：
        # （1）废单或撤单： a—— > 5
        # （2）等待： a—— > 3
        # （3）瞬间成功: a—— > a—— > 0
        # （4）等待一会成功： a—— > 3—— > 3—— > 0
        # （5）部分成交后撤单：a—— > 1—— > 2
        # print('OnRtnOrder')
        # 预埋单：
        # 状态：
        # 撤单： b -- > c -->  5 (VolumeTraded和VolumeTotal都为0)
        # 成功发出： b-- > c --> a
        # 仓位不足： b -- > c
        self.logger.info('==================================CTP系统订单回报====================================')
        self.logger.info('订单状态：' + str(pOrder.OrderStatus))
        self.logger.info('订单引用：' + str(pOrder.OrderRef))
        self.logger.info('订单号：' + str(pOrder.OrderSysID))
        self.logger.info('ContingentCondition：' + str(pOrder.ContingentCondition))
        self.logger.info('报单合约：' + str(pOrder.InstrumentID))
        self.logger.info('RequestID：' + str(pOrder.RequestID))
        self.logger.info('OrderStatus：' + str(pOrder.OrderStatus))
        self.logger.info('VolumeTotal：' + str(pOrder.VolumeTotal))
        self.logger.info('VolumeTraded：' + str(pOrder.VolumeTraded))
        self.logger.info('LimitPrice：' + str(pOrder.LimitPrice))
        # print('OrderLocalID：', str(pOrder.OrderLocalID))
        # print('FrontID：', str(pOrder.FrontID))
        # print('FrontID：', str(pOrder.FrontID))
        # print('OrderType：', str(pOrder.OrderType))
        self.logger.info('OrderSysID：' + str(pOrder.OrderSysID))
        # print('OrderSubmitStatus：', str(pOrder.OrderSubmitStatus))
        self.logger.info('==================================CTP系统订单回报====================================')
        try:
            # 是否为当前进程发出的报单标识
            if pOrder is None:
                return
            if pOrder.ContingentCondition == '1':
                # 不是条件单
                self.ctp_trade_api.init_ctp_order_id(int(pOrder.OrderRef), self.account)

            # ctp订单状态转为平台订单状态
            self.ctp_data_trans_util.order_status_trans(pOrder)
            order_id = str(pOrder.FrontID).strip() + str(pOrder.SessionID).strip() + str(pOrder.OrderRef).strip()
            order = self.ctp_trade_api.trade_adapter.get_old_work_order(self.account, order_id,
                                                                        pOrder.RelativeOrderSysID)

            if order.status == pOrder.status and order.order_sys_id == str(pOrder.OrderSysID).strip():
                self.logger.info('当前进程处理过该订单状态,订单sys_id:{},订单状态:{}'.format(order.order_sys_id,
                                                                           str(pOrder.status)))
                return

            self.ctp_data_trans_util.order_data_trans(pOrder, order)

            # 处理订单回报
            if self.account in self.ctp_trade_api.ctp_qry_thread_dict.keys():
                if pOrder.OrderStatus == api.THOST_FTDC_OST_NoTradeQueueing \
                        or (pOrder.OrderStatus == api.THOST_FTDC_OST_Unknown and pOrder.OrderSysID != ''):
                    # 未知订单回报会回报两次，只要处理第二次有OrderSysID的情况，未成交在队列中是中间状态，不需要处理账号情况
                    pass
                else:
                    self.trade_adapter.on_rtn_order(self.account, order)

                    if pOrder.OrderStatus != api.THOST_FTDC_OST_PartTradedQueueing \
                            and pOrder.OrderStatus != api.THOST_FTDC_OST_AllTraded \
                            or "&" in order.order_book_id:
                        # 当有合约成交时，在成交回报中查询，如何是组合合约，则要都查询,更新账号可用资金和冻结资金
                        self.ctp_trade_api.ctp_qry_thread_dict[self.account].add_event((0, self.account))

            # 处理撤单回报
            if pOrder.OrderStatus == api.THOST_FTDC_OST_Canceled \
                    or pOrder.OrderStatus == api.THOST_FTDC_OST_PartTradedNotQueueing \
                    or pOrder.OrderStatus == api.THOST_FTDC_OST_NoTradeNotQueueing:
                res_mes = '订单撤单, 账号:' + str(self.account) + ',期货' + str(pOrder.InstrumentID) + ',' + \
                          pOrder.StatusMsg + ',订单号:' + str(pOrder.OrderRef)
                print(res_mes)
                if pOrder.StatusMsg == '已撤单' and pOrder.TimeCondition != api.THOST_FTDC_TC_IOC:
                    # 排除手动撤单或者不为当前系统的
                    pass
                else:
                    if pOrder.ContingentCondition == '1':
                        self.trade_adapter.on_system_cancel_order(self.account, order)

        except Exception as e:
            self.logger.error('报单回报异常，错误日志:%s' % str(traceback.format_exc()))

    def OnRspOrderAction(self, pInputOrderAction, pRspInfo, nRequestID, bIsLast):
        self.logger.info('OnRspOrderAction')
        try:
            if pRspInfo is None or pInputOrderAction is None:
                return
            if pRspInfo.ErrorID != 0:
                res_mes = '订单撤单失败,账号:' + str(self.account) + '期货:' + \
                          str(pInputOrderAction.InstrumentID) + '原因:' + str(pRspInfo.ErrorMsg)
                self.logger.info(res_mes)
        except Exception as e:
            self.logger.error('OnRspOrderAction:', str(e))

    def OnRtnTrade(self, pTrade):
        try:
            if pTrade is None:
                return

            order_sys_id = str(pTrade.OrderSysID).strip()
            market = str(pTrade.ExchangeID).strip()
            date = str(str(pTrade.TradeDate).strip())
            trade = XbBacktestTrade()
            self.ctp_data_trans_util.trade_data_trans(pTrade, trade)

            self.logger.info('查询订单order_sys_id:' + str(order_sys_id))
            self.logger.info('查询订单market:' + str(market))
            self.logger.info('查询订单date:' + str(date))
            old_order = self.ctp_trade_api.trade_adapter.get_work_order_by_order_sys_id(self.account, order_sys_id)

            if old_order is not None:
                trade.client_id = old_order.client_id
                trade.order_id = old_order.order_id
                trade.order_remark = old_order.remark
                trade.now_system_order = old_order.now_system_order
            else:
                trade.client_id = '000001'
                trade.now_system_order = 0
                self.logger.info('当前是三方订单成交')

            print("这里耗时1")
            trade.account_id = self.account
            self.trade_adapter.on_rtn_trade(self.account, trade)
            if self.account in self.ctp_trade_api.ctp_qry_thread_dict.keys():
                self.ctp_trade_api.ctp_qry_thread_dict[self.account].add_event((0, self.account))
                zh_list = self.ctp_trade_api.ctp_qry_thread_dict[self.account].is_in_zh_symbol(trade.contract_code)
                if len(zh_list) > 0:
                    # 如果是组合合约中的子合约，则需要刷新所有合约持仓，更新合约的保证金、手续费指标
                    for zh in zh_list:
                        self.ctp_trade_api.ctp_qry_thread_dict[self.account].add_event(
                            (2, self.account, zh, 0))
                self.ctp_trade_api.ctp_qry_thread_dict[self.account].add_event(
                    (1, self.account, trade.contract_code, 0))

        except Exception as e:
            self.logger.error('成交回报异常，错误日志:%s' % str(traceback.format_exc()))

    def OnRspError(self, pRspInfo, nRequestID, bIsLast):
        try:
            self.logger.info('CTP返回错误了,nRequestID:' + str(nRequestID))
            self.logger.info(pRspInfo.ErrorMsg)
            self.logger.info(pRspInfo.ErrorID)
            self.ctp_trade_api.retry_qry(self.account, nRequestID)
        except Exception as e:
            self.logger.error('CTP返回错误异常，错误日志:%s' % str(traceback.format_exc()))

    def OnRspQryInstrument(self, pInstrument, pRspInfo, nRequestID, bIsLast):
        if pInstrument is not None:
            self.logger.info('合约回报')
            xb_back_test_instrument = XbBacktestInstrument()
            self.ctp_data_trans_util.instrument_data_trans(pInstrument, xb_back_test_instrument)
            # self.trade_adapter.on_rsp_qry_instrument(self.account, xb_back_test_instrument)
        if bIsLast:
            self.logger.info('开始查询保证金')
            self.ctp_trade_api.qry_instrument_margin(self.account)

    def OnRspQryInstrumentMarginRate(self, pInstrumentMarginRate, pRspInfo, nRequestID, bIsLast):
        self.logger.info("OnRspQryInstrumentMarginRate")
        self.logger.info(pRspInfo)
        self.logger.info(pInstrumentMarginRate)
        if pInstrumentMarginRate is not None:
            self.logger.info('合约保证金回报:{}'.format(pInstrumentMarginRate.InstrumentID))
            self.logger.info('LongMarginRatioByMoney:{}'.format(pInstrumentMarginRate.LongMarginRatioByMoney))
            self.logger.info('LongMarginRatioByVolume:{}'.format(pInstrumentMarginRate.LongMarginRatioByVolume))
            self.logger.info('ShortMarginRatioByMoney:{}'.format(pInstrumentMarginRate.ShortMarginRatioByMoney))
            self.logger.info('ShortMarginRatioByVolume:{}'.format(pInstrumentMarginRate.ShortMarginRatioByVolume))
            xb_back_test_instrument = XbBacktestInstrument()
            self.ctp_data_trans_util.instrument_margin_data_trans(pInstrumentMarginRate, xb_back_test_instrument)
            # self.trade_adapter.on_rsp_qry_instrument(self.account, xb_back_test_instrument)

    def OnRspUserPasswordUpdate(self, pUserPasswordUpdate, pRspInfo, nRequestID, bIsLast):
        self.logger.info('OnRspUserPasswordUpdate')
        if pRspInfo and pRspInfo.ErrorID != 0:
            self.logger.info('期货账号' + str(self.account) + '修改密码失败，原因：' + str(pRspInfo.ErrorMsg))
            return
        else:
            self.logger.info('期货账号：' + str(self.account) + '修改密码成功')

    def OnRtnFromBankToFutureByBank(self, pRspTransfer):
        self.logger.info('OnRtnFromBankToFutureByBank')
        try:
            xb_r_w_d = XbRealWithdrawDeposit()
            xb_r_w_d.account_id = self.account
            xb_r_w_d.type = 0
            self.ctp_data_trans_util.withdraw_deposit_trans(pRspTransfer, xb_r_w_d)
            self.trade_adapter.on_rtn_transfer(self.account, xb_r_w_d)
            self.ctp_trade_api.ctp_qry_thread_dict[self.account].add_event((0, self.account))
        except Exception as e:
            self.logger.error('OnRtnFromBankToFutureByBank异常，错误日志:%s' % str(traceback.format_exc()))

    def OnRtnFromFutureToBankByBank(self, pRspTransfer):
        self.logger.info('OnRtnFromFutureToBankByBank')
        try:
            xb_r_w_d = XbRealWithdrawDeposit()
            xb_r_w_d.account_id = self.account
            xb_r_w_d.type = 1
            self.ctp_data_trans_util.withdraw_deposit_trans(pRspTransfer, xb_r_w_d)
            self.trade_adapter.on_rtn_transfer(self.account, xb_r_w_d)
            self.ctp_trade_api.ctp_qry_thread_dict[self.account].add_event((0, self.account))
        except Exception as e:
            self.logger.error('OnRtnFromFutureToBankByBank异常，错误日志:%s' % str(traceback.format_exc()))

    def OnRtnFromBankToFutureByFuture(self, pRspTransfer):
        self.logger.info('OnRtnFromBankToFutureByFuture')
        try:
            xb_r_w_d = XbRealWithdrawDeposit()
            xb_r_w_d.account_id = self.account
            xb_r_w_d.type = 0
            self.ctp_data_trans_util.withdraw_deposit_trans(pRspTransfer, xb_r_w_d)
            self.trade_adapter.on_rtn_transfer(self.account, xb_r_w_d)
            self.ctp_trade_api.ctp_qry_thread_dict[self.account].add_event((0, self.account))
        except Exception as e:
            self.logger.error('OnRtnFromBankToFutureByFuture异常，错误日志:%s' % str(traceback.format_exc()))

    def OnRtnFromFutureToBankByFuture(self, pRspTransfer):
        self.logger.info('出金')
        try:
            xb_r_w_d = XbRealWithdrawDeposit()
            xb_r_w_d.account_id = self.account
            xb_r_w_d.type = 1
            self.ctp_data_trans_util.withdraw_deposit_trans(pRspTransfer, xb_r_w_d)
            self.trade_adapter.on_rtn_transfer(self.account, xb_r_w_d)
            self.ctp_trade_api.ctp_qry_thread_dict[self.account].add_event((0, self.account))
        except Exception as e:
            self.logger.error('OnRtnFromFutureToBankByFuture异常，错误日志:%s' % str(traceback.format_exc()))
