import os
import threading
from utils.data.symbol_util import SymbolUtil
from utils.log.log_factory import LogFactory

from panda_backtest.backtest_common.constant.strategy_constant import *
from panda_backtest.backtest_common.constant.string_constant import FUTURE_ACCOUNT_NOT_INIT, CTP_INSERT_ORDER_ERROR, \
    FUTURE_ORDER_ID_NOT_INIT
from panda_trading.real_trade_api.ctp.ctp_trade_spi import CTPTraderSpi
from panda_trading.real_trade_api.ctp.dur_result import DurResult
from panda_trading.real_trade_api.ctp.qry_thread import QryThread
import platform
import ctp as api


class CTPTradeApi(object):
    ACCOUNT_NOT_FOUND = 100

    dur_result = DurResult()

    def __init__(self, mock_id, trade_adapter):
        # 保存登录账号的CTP操作API
        self.account_trade_api_dict = {}
        self.account_trade_spi_dict = {}
        # 保存当前下单序号
        self.order_id_dict = dict()
        # 保存成功登录后用户登录资料,用于重新登录
        self.account_info_dicts = dict()
        self.trade_adapter = trade_adapter
        self.req_id = 10
        self.mock_id = mock_id
        self.ctp_qry_thread_dict = dict()
        self.req_history = dict()
        self.order_lock = dict()
        self.logger = LogFactory.get_logger()

    def init_account_ctp_qry_thread(self, account):
        ctp_qry_thread = QryThread(self.query_account, self.query_positions, self.query_combine_positions,
                                   self.resume_login)
        self.ctp_qry_thread_dict[account] = ctp_qry_thread

    def start_account_ctp_qry_thread(self, account):
        """
        启动ctp连接查询监控线程
        :param account:
        :return:
        """
        self.ctp_qry_thread_dict[account].start_qry()
        self.ctp_qry_thread_dict[account].check_account_con(account)

    def init_ctp_account_info(self, account):
        """
        初始化第一次查询账号持仓资金情况
        :param account:
        :return:
        """
        self.ctp_qry_thread_dict[account].add_event((0, account))
        self.ctp_qry_thread_dict[account].add_event((1, account))
        self.ctp_qry_thread_dict[account].add_event((2, account))

    def init_ctp_order_id(self, init_id, account):
        if account in self.order_id_dict.keys():
            self.order_id_dict[account] = max(self.order_id_dict[account], init_id)
        else:
            self.order_id_dict[account] = init_id

    def init_account_front_session(self, account, front_id, session_id):
        if account in self.account_info_dicts.keys():
            account_info = self.account_info_dicts[account]
            account_info['front_id'] = front_id
            account_info['session_id'] = session_id

    def init_ctp_trade_api(self, ip, port, account, password, broker_id, auth_code, app_id=None):
        if account in self.account_trade_api_dict.keys():
            self.logger.info('ctp交易账号已经登录，账号：' + account)
            # TODO：杰宜斯注释
            # ctp_trade_api = self.account_trade_api_dict[account]
            # ctp_trade_api.Init()
            return

        if auth_code is None:
            is_auth_flag = 0
        else:
            is_auth_flag = 1
        log_dir = os.environ.get("LOG_DIR")
        # CTP最新封装支持 Darwin的 cpu arm64 m1平台
        if log_dir is None:
            if platform.system() in ('Linux', 'Darwin'):
                log_dir = '/tmp/sunrise/ctp/'
            else:
                log_dir = 'C:\\sunrise\\ctp\\'
        log_dir = log_dir + account + os.sep
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        account_info_dict = dict()
        account_info_dict['ip'] = ip
        account_info_dict['port'] = port
        account_info_dict['broker_id'] = broker_id
        account_info_dict['account'] = account
        account_info_dict['password'] = password
        account_info_dict['auth_code'] = auth_code
        if app_id is None:
            account_info_dict['app_id'] = 'ver'
        else:
            account_info_dict['app_id'] = app_id
        self.account_info_dicts[account] = account_info_dict
        self.order_lock[account] = threading.Lock()

        ctp_trade_api = api.CThostFtdcTraderApi_CreateFtdcTraderApi(log_dir)
        ctp_trade_spi = CTPTraderSpi(self, account, is_auth_flag)
        # 需要将回调实体实例化，否则ctp会直接退出
        self.account_trade_api_dict[account] = ctp_trade_api
        # CTPTradeApi.dur_result.save_spi(account, ctp_trade_spi)
        self.account_trade_spi_dict[account] = ctp_trade_spi
        ctp_trade_api.RegisterSpi(ctp_trade_spi)
        # ctp_trade_api.SubscribePrivateTopic(api.THOST_TERT_RESTART)
        # ctp_trade_api.SubscribePublicTopic(api.THOST_TERT_RESTART)
        ctp_trade_api.SubscribePrivateTopic(api.THOST_TERT_QUICK)
        ctp_trade_api.SubscribePublicTopic(api.THOST_TERT_QUICK)
        ctp_trade_api.RegisterFront(ip)
        ctp_trade_api.Init()

    def account_login(self, account):
        """账号登录"""

        account_info = self.account_info_dicts[account]

        login_field = api.CThostFtdcReqUserLoginField()
        login_field.BrokerID = str(account_info['broker_id'])
        login_field.UserID = str(account)
        login_field.Password = str(account_info['password'])
        login_field.UserProductInfo = 'redefine'
        if account in self.account_trade_api_dict.keys():
            ctp_trade_api = self.account_trade_api_dict[account]
            res = ctp_trade_api.ReqUserLogin(login_field, self.req_id)
            self.req_id += 1

            if res != 0:
                print('ctp交易账号登录失败，账号：' + account + ',错误码:' + res)
                del self.account_trade_api_dict[account]
            else:
                self.req_history.clear()
                self.req_id = 10

    def account_auth(self, account):
        account_info = self.account_info_dicts[account]
        auth_field = api.CThostFtdcReqAuthenticateField()
        auth_field.BrokerID = str(account_info['broker_id'])
        auth_field.UserID = str(account)
        auth_field.UserProductInfo = 'redefine'
        auth_field.AuthCode = str(account_info['auth_code'])
        auth_field.AppID = str(account_info['app_id'])

        if account in self.account_trade_api_dict.keys():
            ctp_trade_api = self.account_trade_api_dict[account]
            res = ctp_trade_api.ReqAuthenticate(auth_field, self.req_id)
            self.req_id += 1

            if res != 0:
                print('ctp交易账号认证失败，账号：' + account + ',错误码:' + res)
                del self.account_trade_api_dict[account]

    def account_settle_confirm(self, account):
        account_info = self.account_info_dicts[account]

        settle_confirm_field = api.CThostFtdcSettlementInfoConfirmField()
        settle_confirm_field.BrokerID = str(account_info['broker_id'])
        settle_confirm_field.InvestorID = str(account)

        if account in self.account_trade_api_dict.keys():
            ctp_trade_api = self.account_trade_api_dict[account]
            res = ctp_trade_api.ReqSettlementInfoConfirm(settle_confirm_field, self.req_id)
            self.req_id += 1
            if res != 0:
                self.logger.error('ctp交易账号结算失败，账号：' + account + ',错误码:' + res)

    def account_logout(self, account):
        if account in self.account_trade_api_dict.keys():
            self.req_history.clear()
            ctp_trade_api = self.account_trade_api_dict[account]
            account_info = self.account_info_dicts[account]
            broker_id = account_info['broker_id']

            logout_field = api.CThostFtdcUserLogoutField()
            logout_field.BrokerID = str(broker_id)
            logout_field.UserID = str(account)

            res = ctp_trade_api.ReqUserLogout(logout_field, self.req_id)
            self.req_id += 1
            if res == 0:
                self.logger.info('ctp交易账号登出成功，账号：' + account)
                # TODO：杰宜斯需要注释
                ctp_trade_api.Release()
                del self.account_trade_api_dict[account]
            else:
                self.logger.error('ctp交易账号登出失败，账号：{}, 错误码：{}'.format(str(account), str(res)))
        else:
            self.logger.error('ctp交易账号登出失败，账号不存在，账号：' + account)

    def resume_login(self, account):
        self.logger.info('resume_login')
        if account in self.account_trade_api_dict.keys():
            ctp_trade_api = self.account_trade_api_dict[account]
            # TODO：杰宜斯需要注释
            self.logger.info("期货柜台连接销毁")
            ctp_trade_api.Release()
            del self.account_trade_api_dict[account]
        if account in self.account_info_dicts.keys():
            account_info = self.account_info_dicts[account]
            self.init_ctp_trade_api(
                account_info['ip'],
                account_info['port'],
                account_info['account'],
                account_info['password'],
                account_info['broker_id'],
                account_info['auth_code'],
                account_info['app_id'])
        else:
            self.logger.error('ctp账号重新登录失败，账号：' + account)

    def query_account(self, account):
        """查询账号"""
        self.logger.info("查询账号")
        if account in self.account_trade_api_dict.keys() and 'session_id' in self.account_info_dicts[account].keys():
            ctp_trade_api = self.account_trade_api_dict[account]
            account_info = self.account_info_dicts[account]
            broker_id = account_info['broker_id']

            qry_account_field = api.CThostFtdcQryTradingAccountField()
            qry_account_field.BrokerID = str(broker_id)
            qry_account_field.InvestorID = str(account)

            self.req_history[self.req_id] = (0, account)
            res = ctp_trade_api.ReqQryTradingAccount(qry_account_field, self.req_id)
            self.req_id = self.req_id + 1

            if res == 0:
                return res
            else:
                self.logger.error("账号查询失败，账号：" + account + "；错误码：" + str(res))
                return res
        else:
            self.logger.error("账号查询失败，账号：" + account + "；原因：账号不存在或未登录")
            return self.ACCOUNT_NOT_FOUND

    def query_positions(self, account, symbol=None, req_id=None):
        """查询账号"""
        self.logger.info("发出持仓查询")
        if account in self.account_trade_api_dict.keys():
            ctp_trade_api = self.account_trade_api_dict[account]
            account_info = self.account_info_dicts[account]
            broker_id = account_info['broker_id']

            qry_position_field = api.CThostFtdcQryInvestorPositionField()
            qry_position_field.BrokerID = str(broker_id)
            qry_position_field.InvestorID = str(account)
            if symbol is not None:
                qry_position_field.InstrumentID = str(symbol)

            if req_id is None:
                req_id = self.req_id

            self.req_history[req_id] = (1, account, symbol, req_id)
            res = ctp_trade_api.ReqQryInvestorPosition(qry_position_field, req_id)
            self.req_id += 1
            if res == 0:
                return res
            else:
                self.logger.error("持仓查询失败，账号：" + account + "；错误码：" + str(res))
                return res
        else:
            self.logger.error("持仓查询失败，账号：" + account + "；原因：账号不存在或未登录")
            return self.ACCOUNT_NOT_FOUND

    def query_combine_positions(self, account, symbol=None, req_id=None):
        """查询组合持仓"""
        self.logger.info("发出组合持仓查询")
        if account in self.account_trade_api_dict.keys():
            ctp_trade_api = self.account_trade_api_dict[account]
            account_info = self.account_info_dicts[account]
            broker_id = account_info['broker_id']

            qry_position_field = api.CThostFtdcQryInvestorPositionCombineDetailField()
            qry_position_field.BrokerID = str(broker_id)
            qry_position_field.InvestorID = str(account)
            if symbol is not None:
                qry_position_field.CombInstrumentID = str(symbol)

            if req_id is None:
                req_id = self.req_id

            self.req_history[req_id] = (2, account, symbol, req_id)
            res = ctp_trade_api.ReqQryInvestorPositionCombineDetail(qry_position_field, req_id)
            self.req_id += 1
            if res == 0:
                return res
            else:
                self.logger.error("组合持仓查询失败，账号：" + account + "；错误码：" + str(res))
                return res
        else:
            self.logger.error("组合持仓查询失败，账号：" + account + "；原因：账号不存在或未登录")
            return self.ACCOUNT_NOT_FOUND

    def query_positions_detail(self, account, symbol=None, req_id=None):
        """查询账号"""
        if account in self.account_trade_api_dict.keys():
            ctp_trade_api = self.account_trade_api_dict[account]
            account_info = self.account_info_dicts[account]
            broker_id = account_info['broker_id']

            qry_position_field = api.CThostFtdcQryInvestorPositionDetailField()
            qry_position_field.BrokerID = str(broker_id)
            qry_position_field.InvestorID = str(account)
            if symbol is not None:
                qry_position_field.InstrumentID = str(symbol)

            if req_id is None:
                req_id = self.req_id

            self.req_history[req_id] = (1, account, symbol, req_id)
            res = ctp_trade_api.ReqQryInvestorPositionDetail(qry_position_field, req_id)
            self.req_id += 1
            if res == 0:
                return res
            else:
                self.logger.error("持仓查询失败，账号：" + account + "；错误码：" + str(res))
                return res
        else:
            self.logger.error("持仓查询失败，账号：" + account + "；原因：账号不存在或未登录")
            return self.ACCOUNT_NOT_FOUND

    def insert_order(self, account, order_result):
        """下单"""
        if account in self.account_trade_api_dict.keys() and account in self.account_info_dicts.keys() \
                and 'front_id' in self.account_info_dicts[account].keys():
            ctp_trade_api = self.account_trade_api_dict[account]
            account_info = self.account_info_dicts[account]

            # 交易报价实体
            order_field = api.CThostFtdcInputOrderField()

            order_field.BrokerID = str(account_info['broker_id'])
            order_field.InvestorID = str(account)
            order_field.UserID = str(account)
            symbol = SymbolUtil.symbol_to_ctp_code(order_result.order_book_id)
            order_field.ExchangeID = order_result.market
            order_field.InstrumentID = str(symbol)
            order_field.TimeCondition = api.THOST_FTDC_TC_IOC
            if order_result.side == SIDE_BUY:
                order_field.Direction = api.THOST_FTDC_D_Buy
            else:
                order_field.Direction = api.THOST_FTDC_D_Sell

            if order_result.effect == OPEN:
                order_field.CombOffsetFlag = api.THOST_FTDC_OF_Open
            else:
                if order_result.is_td_close == 0:
                    order_field.CombOffsetFlag = api.THOST_FTDC_OF_Close
                else:
                    order_field.CombOffsetFlag = api.THOST_FTDC_OF_CloseToday

            order_field.CombHedgeFlag = api.THOST_FTDC_HF_Speculation
            order_field.VolumeTotalOriginal = order_result.quantity
            order_field.MinVolume = 0
            order_field.VolumeCondition = api.THOST_FTDC_VC_AV
            order_field.ForceCloseReason = api.THOST_FTDC_FCC_NotForceClose
            order_field.IsAutoSuspend = 0
            order_field.UserForceClose = 0
            order_field.ContingentCondition = api.THOST_FTDC_CC_Immediately

            # 加同步锁，防止多线程报单号重复
            mutex = self.order_lock[account]
            try:
                mutex.acquire(timeout=30)

                if account in self.order_id_dict.keys():
                    order_id = self.order_id_dict[account] + 1
                    self.order_id_dict[account] = order_id
                else:
                    order_result.message = FUTURE_ORDER_ID_NOT_INIT
                    order_result.status = REJECTED
                    self.trade_adapter.on_system_cancel_order(account, order_result)
                    return order_result

                order_ref = str(self.order_id_dict[account])
                order_field.OrderRef = str(order_ref)
                # order_field.UserID = req.get('client_id', self.mock_id)

                if order_result.price_type == LIMIT:
                    order_field.OrderPriceType = api.THOST_FTDC_OPT_LimitPrice
                    order_field.LimitPrice = order_result.price
                    order_field.TimeCondition = api.THOST_FTDC_TC_IOC
                else:
                    order_field.OrderPriceType = api.THOST_FTDC_OPT_AnyPrice
                    order_field.LimitPrice = 0
                    order_field.TimeCondition = api.THOST_FTDC_TC_IOC

                order_result.order_id = str(account_info['front_id']) + str(account_info['session_id']) + str(order_ref)
                order_result.order_client_id = order_ref
                order_result.status = WAIT
                # self.trade_adapter.on_update_local_order(account, order_result)
                self.trade_adapter.on_rtn_order(account, order_result)
                res = ctp_trade_api.ReqOrderInsert(order_field, self.req_id)
                self.req_id += 1
                if res == 0:
                    return order_result
                else:
                    order_result.message = CTP_INSERT_ORDER_ERROR.format(res)
                    order_result.status = REJECTED
                    self.trade_adapter.on_system_cancel_order(account, order_result)
                    return order_result
            finally:
                mutex.release()

        else:
            order_result.message = FUTURE_ACCOUNT_NOT_INIT
            order_result.status = REJECTED
            self.trade_adapter.on_system_cancel_order(account, order_result)
            return order_result

    def cancel_order(self, account, order_id):
        """撤单"""
        if account in self.account_trade_api_dict.keys():
            order = self.trade_adapter.get_work_order_by_order_id(account, order_id)
            if order is not None:

                ctp_trade_api = self.account_trade_api_dict[account]
                account_info = self.account_info_dicts[account]

                req = api.CThostFtdcInputOrderActionField()
                req.BrokerID = str(account_info['broker_id'])
                req.InvestorID = str(account)
                req.UserID = str(account)
                req.ExchangeID = str(order.market)
                # req.InstrumentID = str(order.order_book_id)
                req.OrderRef = str(order.order_client_id)
                req.OrderSysID = str(order.order_sys_id)
                req.ActionFlag = api.THOST_FTDC_AF_Delete

                res = ctp_trade_api.ReqOrderAction(req, self.req_id)
                self.req_id += 1
                if res == 0:
                    return True
                else:
                    self.logger.error("撤单失败，账号：" + account + "；错误码：" + str(res))
                    return False
            else:
                self.logger.error("撤单失败，账号：" + account + "；订单不存在")
                return False

        else:
            self.logger.error("撤单失败，账号：" + account + "；原因：账号不存在或未登录")
            return False

    def get_trading_day(self):
        return None

    def qry_instrument(self, account):

        if account in self.account_trade_api_dict.keys():
            ctp_trade_api = self.account_trade_api_dict[account]
            qry_instrument_field = api.CThostFtdcQryInstrumentField()
            res = ctp_trade_api.ReqQryInstrument(qry_instrument_field, self.req_id)
            self.req_id += 1
            if res == 0:
                return True
            else:
                self.logger.error("查询合约失败，账号：" + account + "；错误码：" + str(res))
                return False
        else:
            self.logger.error("查询合约失败，账号：" + account + "；原因：账号不存在或未登录")

    def qry_instrument_margin(self, account):
        if account in self.account_trade_api_dict.keys():
            print('发出qry_instrument_margin查询')
            ctp_trade_api = self.account_trade_api_dict[account]
            qry_instrument_field = api.CThostFtdcQryInstrumentMarginRateField()
            account_info = self.account_info_dicts[account]
            broker_id = account_info['broker_id']
            qry_instrument_field.BrokerID = broker_id
            qry_instrument_field.InvestorID = account
            qry_instrument_field.HedgeFlag = api.THOST_FTDC_HF_Speculation
            # qry_instrument_field.InstrumentID = 'bb21'
            # qry_instrument_field.ExchangeID = api.THOST_FTDC_EIDT_SHFE
            res = ctp_trade_api.ReqQryInstrumentMarginRate(qry_instrument_field, self.req_id)
            self.req_id += 1
            if res == 0:
                return True
            else:
                self.logger.error("查询合约保证金失败，账号：" + account + "；错误码：" + str(res))
                return False
        else:
            self.logger.error("查询合约保证金失败，账号：" + account + "；原因：账号不存在或未登录")

    def req_user_password_update(self, account):
        if account in self.account_trade_api_dict.keys():
            ctp_trade_api = self.account_trade_api_dict[account]
            update_pwd_field = api.CThostFtdcUserPasswordUpdateField()
            account_info = self.account_info_dicts[account]
            broker_id = account_info['broker_id']
            update_pwd_field.BrokerID = broker_id
            update_pwd_field.UserID = account
            update_pwd_field.OldPassword = '111111'
            update_pwd_field.NewPassword = 'xb123456'
            res = ctp_trade_api.ReqUserPasswordUpdate(update_pwd_field, self.req_id)
            self.req_id += 1
            if res == 0:
                self.logger.info("修改密码成功，账号：" + account)
                return True
            else:
                self.logger.error("修改密码失败，账号：" + account + "；错误码：" + str(res))
                return False
        else:
            self.logger.error("修改密码失败，账号：" + account + "；原因：账号不存在或未登录")

    def get_ctp_account_info(self, account):
        if account in self.account_info_dicts.keys():
            return self.account_info_dicts[account]
        else:
            return None

    def retry_qry(self, account, req_id):
        if req_id in self.req_history.keys():
            self.logger.error('ctp错误重新查询：%s' % str(self.req_history[req_id]))
            self.ctp_qry_thread_dict[account].add_event(self.req_history[req_id])
