import datetime

class TimeUtil:

    @staticmethod
    def get_time_range(start_time, end_time):
        """ 查看起始時間和結束時間之差 """
        start_time = datetime.datetime.strptime(start_time, '%Y%m%d')
        end_time = datetime.datetime.strptime(end_time, '%Y%m%d')
        time_range = (end_time - start_time).days
        return time_range