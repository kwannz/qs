
import abc
import logging

from six import with_metaclass

class OrderVerify(with_metaclass(abc.ABCMeta)):

    @abc.abstractmethod
    def can_submit_order(self, account, order_result):
        pass
