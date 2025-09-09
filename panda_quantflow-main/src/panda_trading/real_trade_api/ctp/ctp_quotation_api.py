import os
from sys import platform

import logging
from panda_trading.real_trade_api.ctp.ctp_quotation_spi import CtpQuotationSpi
from panda_trading.real_trade_api.ctp.dur_result import DurResult
from panda_trading.real_trade_api.ctp.quotation_qry_thread import QuotationQryThread
import ctp as mdapi

from utils.data.symbol_util import SymbolUtil


class CtpQuotationApi(object):

    ACCOUNT_NOT_FOUND = 100
    dur_result = DurResult()

    def __init__(self, mock_id, trade_adapter):
        super(CtpQuotationApi, self).__init__()
        self.trade_adapter = trade_adapter
        self.mock_id = mock_id
        self.ctp_quotation_spi = None
        self.ctp_quotation_api = None
        # 保存成功登录后用户登录资料,用于重新登录
        self.account_info_dicts = dict()
        self.req_id = 1
        self.xtp_qry_thread_dict = dict()
        self.sub_symbol_list = set()
        self.quotation_qry_thread = QuotationQryThread(self.resume_login)
        self.logger = logging.getLogger(__name__)

    def init_ctp_quotation_api(self, ip, port, account, password, broker_id):
        if self.ctp_quotation_api is None:
            log_dir = os.environ.get("LOG_DIR")
            # CTP最新封装支持 Darwin的 cpu arm64 m1平台
            if log_dir is None:
                if platform.system() == 'Linux':
                    log_dir = '/usr/sunrise/ctp/quotation/'
                    log_dir = log_dir + str(self.mock_id) + os.sep
                else:
                    log_dir = 'C:\\sunrise\\ctp\\quotation\\'
            log_dir = log_dir + str(self.mock_id) + os.sep
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            account_info_dict = dict()
            account_info_dict['ip'] = ip
            account_info_dict['port'] = port
            account_info_dict['account'] = account
            account_info_dict['password'] = password
            account_info_dict['broker_id'] = broker_id
            self.account_info_dicts[account] = account_info_dict

            ctp_quotation_spi = CtpQuotationSpi(self, account)
            self.ctp_quotation_spi = ctp_quotation_spi
            self.ctp_quotation_api = mdapi.CThostFtdcMdApi_CreateFtdcMdApi()
            self.ctp_quotation_api.RegisterFront(ip)
            self.ctp_quotation_api.RegisterSpi(self.ctp_quotation_spi)
            self.ctp_quotation_api.Init()

    def account_login(self, account):
        self.sub_symbol_list.clear()
        account_info = self.account_info_dicts[account]
        loginfield = mdapi.CThostFtdcReqUserLoginField()
        loginfield.BrokerID = str(account_info['broker_id'])
        loginfield.UserID = str(account)
        loginfield.Password = str(account_info['password'])
        loginfield.UserProductInfo = "python dll"
        res = self.ctp_quotation_api.ReqUserLogin(loginfield, 0)
        if res != 0:
            self.logger.info('ctp交易账号登录失败，账号：' + account + ',错误码:' + res)

    def init_account_ctp_quotation_qry_thread(self, account):
        self.quotation_qry_thread.check_account_con(account)

    def account_logout(self, account):
        if account in self.account_info_dicts.keys():
            account_info = self.account_info_dicts[account]
            login_out_field = mdapi.CThostFtdcUserLogoutField()
            login_out_field.BrokerID = str(account_info['broker_id'])
            login_out_field.UserID = str(account)
            res = self.ctp_quotation_api.ReqUserLogout(login_out_field, 0)
            if res == 0:
                self.logger.info('ctp行情账号登出成功，账号：' + account)
            else:
                self.logger.info('ctp行情账号登出失败，账号：' + account)
        else:
            self.logger.info('ctp行情账号登出失败，账号：' + account + '未在登录状态')

    def resume_login(self, account):
        if account in self.account_info_dicts.keys():
            self.account_logout(account)
        if account in self.account_info_dicts.keys():
            account_info = self.account_info_dicts[account]
            self.init_ctp_quotation_api(
                account_info['ip'],
                account_info['port'],
                account_info['account'],
                account_info['password'],
                account_info['broker_id'])
        else:
            self.logger.info('xtp账号重新登录失败，账号：'+account)

    def sub_symbol(self, symbol_list):
        if self.ctp_quotation_api is None:
            return
        future_code_list = list()
        for future_code in symbol_list:
            future_code = str.encode(SymbolUtil.symbol_to_ctp_code(future_code))
            if future_code not in self.sub_symbol_list:
                future_code_list.append(future_code)
                self.sub_symbol_list.add(future_code)
        if len(future_code_list) > 0:
            self.logger.info('发出新的行情订阅==========》')
            self.logger.info(future_code_list)
            self.ctp_quotation_api.SubscribeMarketData(future_code_list, len(future_code_list))

