import queue
import threading
import time
import traceback

from ..common.set_queue import SetQueue

from utils.data.symbol_util import SymbolUtil
from utils.thread.thread_util import ThreadUtil
from utils.time.time_util import TimeUtil
from panda_trading.real_trade_api.ctp.data.trade_date_data import TradeDateData


class QryThread(object):
    def __init__(self, qry_account, qry_position, qry_position_combine, resume_login):
        self.qry_queue = SetQueue()
        self.qry_account = qry_account
        self.qry_position = qry_position
        self.qry_position_combine = qry_position_combine
        self.resume_login = resume_login
        self.event = threading.Event()
        self.event.set()
        self.con_status = False
        # 保存账号组合套利合约
        self.zh_symbol_list = set()
        self.trade_date_data = TradeDateData()

    def save_zh_symbol(self, zh_symbol):
        self.zh_symbol_list.add(zh_symbol)

    def del_zh_symbol(self, zh_symbol):
        if zh_symbol in self.zh_symbol_list:
            self.zh_symbol_list.remove(zh_symbol)

    def is_in_zh_symbol(self, symbol):
        ctp_symbol = SymbolUtil.symbol_to_ctp_code(symbol)
        zh_list = list()
        for zh_symbol in self.zh_symbol_list:
            if ctp_symbol in zh_symbol:
                zh_list.append(zh_symbol)
        return zh_list

    def check_account_con(self, account):
        time.sleep(5)
        ThreadUtil.hand_cycle_thread(self.start_check_account, (account,), 30)

    def start_check_account(self, account):
        if TimeUtil.in_time_range('084500-113000,130000-151500,000000-023000,204500-235959'):
            if self.con_status is False:
                trade_date_status = self.trade_date_data.is_trade_date(time.strftime("%Y%m%d"))
                if trade_date_status:
                    print('检查到期货交易账号未登陆')
                    self.resume_login(account)

    def change_con_status(self, account, status):
        self.con_status = status
        if status:
            self.add_event((0, account))
            self.add_event((1, account))
            self.add_event((2, account))

    def get_con_status(self):
        return self.con_status

    def qry_end(self):
        self.event.set()

    def add_event(self, qry_item):
        self.qry_queue.put(qry_item)

    def start_qry(self):
        sub_thread = threading.Thread(target=self.qyr_from_queue, name="订阅线程", args=())
        # 创建线程完毕之后，一定要启动
        sub_thread.setDaemon(True)
        sub_thread.start()

    def qyr_from_queue(self):
        while True:
            try:
                qry_item = self.qry_queue.get()
                self.event.wait(50)
                self.event.clear()
                qry_item_type = qry_item[0]
                qry_account = qry_item[1]
                retry = 0
                if qry_item_type == 0:
                    # 查询账号资金信息
                    res = -1
                    while res != 0 and retry < 5:
                        res = self.qry_account(qry_account)
                        # res = 0
                        retry = retry + 1
                        if res != 0:
                            time.sleep(1)

                elif qry_item_type == 1:
                    # 查询持仓
                    res = -1
                    while res != 0 and retry < 5:
                        if len(qry_item) < 4:
                            # 查询账号所有持仓
                            res = self.qry_position(qry_account)
                        else:
                            # 查询指定合约持仓相关信息
                            if '&' in qry_item[2]:
                                res = self.qry_position(qry_account, qry_item[2],
                                                        qry_item[3])
                            else:
                                res = self.qry_position(qry_account, SymbolUtil.symbol_to_ctp_code(qry_item[2]),
                                                        qry_item[3])
                        retry = retry + 1
                        if res != 0:
                            time.sleep(1)
                elif qry_item_type == 2:
                    res = -1
                    while res != 0 and retry < 5:
                        if len(qry_item) < 4:
                            # 查询账号所有组合持仓
                            res = self.qry_position_combine(qry_account)
                        else:
                            res = self.qry_position_combine(qry_account, qry_item[2], qry_item[3])
                        retry = retry + 1
                        if res != 0:
                            time.sleep(1)

            except queue.Empty:
                print('ctp查询线程超时')
                break
            except Exception as e:
                mes = traceback.format_exc()
                print('ctp查询线程异常处理，原因：%s' % str(mes))
