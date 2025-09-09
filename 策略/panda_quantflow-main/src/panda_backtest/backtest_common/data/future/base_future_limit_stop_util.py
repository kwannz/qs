import abc
import logging

from six import with_metaclass

class BaseFutureLimitStopUtil(with_metaclass(abc.ABCMeta)):

    @abc.abstractmethod
    def decide_future_limit(self, future, trade_date, trad_price, side):
        """
        判断期货涨跌停
        :param future:
        :param trade_date:
        :param trad_price:
        :param side:
        :return:
        """
        pass
