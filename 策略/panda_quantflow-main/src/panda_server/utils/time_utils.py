from datetime import datetime, timezone, timedelta
import logging

import pytz

def get_beijing_time() -> datetime:
    """
    获取当前北京时区的时间
    返回：datetime 对象，表示当前北京时间（不带时区信息）
    """
    # 获取北京时区
    beijing_tz = pytz.timezone('Asia/Shanghai')
    # 获取当前 UTC 时间并转换为北京时间
    utc_now = datetime.now(pytz.UTC)
    beijing_time = utc_now.astimezone(beijing_tz)
    # 返回不带时区信息的北京时间
    return beijing_time.replace(tzinfo=None)

def convert_to_beijing_time(dt: datetime) -> datetime:
    """将UTC时间转换为东八区时间"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    beijing_tz = timezone(timedelta(hours=8))
    return dt.astimezone(beijing_tz) 