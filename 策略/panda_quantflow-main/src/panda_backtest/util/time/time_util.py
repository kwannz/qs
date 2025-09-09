import calendar
import time
from dateutil import relativedelta
import datetime
import os

class TimeUtil:

    @staticmethod
    def date_param_handler(date):
        try:
            if isinstance(date, datetime.datetime):
                return date
            elif isinstance(date, str):
                if date.count('-') == 2:
                    return datetime.datetime.strptime(date, "%Y-%m-%d")
                elif date.count('/') == 2:
                    return datetime.datetime.strptime(date, "%Y/%m/%d")
                else:
                    return datetime.datetime.strptime(date, "%Y%m%d")
            elif isinstance(date, int):
                return datetime.datetime.strptime(str(date), "%Y%m%d")
            return None
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def get_report_date(end_date, count):

        report_list = ['0331', '0630', '0930', '1231']

        month_day = end_date.strftime("%m%d")

        cur = 0
        for i in range(len(report_list)):
            if month_day <= report_list[i]:
                break
            else:
                cur = cur + 1

        year_num = int(count / len(report_list))
        ye_num = count % len(report_list)

        if ye_num <= cur:
            res = cur - ye_num
        else:
            year_num = year_num + 1
            res = len(report_list) - (ye_num - cur)

        start_year = end_date + relativedelta.relativedelta(years=-year_num)
        start_year_str = start_year.strftime("%Y")
        start_date_str = start_year_str + report_list[res]
        return datetime.datetime.strptime(start_date_str, '%Y%m%d')

    @staticmethod
    def get_trade_time(x, y):
        y = str(y).zfill(6)
        trade_time = str(x) + y
        return int(trade_time)

    @staticmethod
    def in_time_range(ranges):
        now = time.strptime(time.strftime("%H%M%S"), "%H%M%S")
        ranges = ranges.split(",")
        for range in ranges:
            r = range.split("-")
            if time.strptime(
                r[0], "%H%M%S") <= now <= time.strptime(
                r[1], "%H%M%S") or time.strptime(
                r[0], "%H%M%S") >= now >= time.strptime(
                    r[1], "%H%M%S"):
                return True
        return False

    @staticmethod
    def month_to_time(year_month):
        return datetime.datetime.strptime(str(year_month), "%Y%m")

    @staticmethod
    def get_next_monday():
        today = datetime.date.today()
        oneday = datetime.timedelta(days=1)
        m1 = calendar.MONDAY
        while today.weekday() != m1:
            today += oneday
        next_monday = today.strftime('%Y%m%d')

        return next_monday

    @staticmethod
    def get_last_date(trade_date):
        """
        获取上一个自然日
        :param trade_date:
        :return:
        """
        trade_date_time = datetime.datetime.strptime(trade_date, '%Y%m%d')
        date = trade_date_time + datetime.timedelta(days=-1)  # 2015-10-29 00:00:00
        return date.strftime('%Y%m%d')

    @staticmethod
    def get_begin_to_end_date_list(begin_date, end_date):
        # 前闭后闭
        date_list = []
        begin_date = datetime.datetime.strptime(begin_date, "%Y%m%d")
        end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
        while begin_date <= end_date:
            date_str = begin_date.strftime("%Y%m%d")
            date_list.append(date_str)
            begin_date += datetime.timedelta(days=1)
        return date_list

    @staticmethod
    def datetime_to_utc(trade_time):
        return datetime.datetime.utcfromtimestamp(trade_time.timestamp())

    @staticmethod
    def utc_to_datetime(utc_time):
        local_tm = datetime.datetime.fromtimestamp(0)
        utc_tm = datetime.datetime.utcfromtimestamp(0)
        offset = local_tm - utc_tm
        return utc_time + offset

    @staticmethod
    def update_system_time(to_date, to_time):
        # 设定日期
        _date = datetime.datetime.strptime(to_date, "%Y%m%d")
        # 设定时间为 0点30分
        _time = '00.00.59'
        # 设定时间
        os.system('time {}'.format(to_time))
        os.system('date {}'.format(_date))

# if __name__ == '__main__':
#     res = TimeUtil.in_time_range('093000-113000,130000-150000')
#     print(res)
