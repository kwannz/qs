import time

from .data.trade_date_data import TradeDateData
from utils.thread.thread_util import ThreadUtil
from utils.time.time_util import TimeUtil


class QuotationQryThread(object):
    def __init__(self, resume_login):
        self.con_status = False
        self.resume_login = resume_login
        self.trade_date_data = TradeDateData()

    def check_account_con(self, account):
        time.sleep(5)
        # self.start_check_account(account)
        ThreadUtil.hand_cycle_thread(self.start_check_account, (account,), 30)

    def start_check_account(self, account):
        if TimeUtil.in_time_range('084500-113000,130000-151500,000000-023000,204500-235959'):
            if self.con_status is False:
                trade_date_status = self.trade_date_data.is_trade_date(time.strftime("%Y%m%d"))
                if trade_date_status:
                    print('检查到账号未登陆')
                    self.resume_login(account)

    def change_con_status(self, status):
        self.con_status = status

    def get_con_status(self):
        return self.con_status