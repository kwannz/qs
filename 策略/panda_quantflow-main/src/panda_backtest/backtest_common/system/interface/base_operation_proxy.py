import abc
import logging

from six import with_metaclass

class BaseOperationProxy(with_metaclass(abc.ABCMeta)):

    @abc.abstractmethod
    def get_account_info(self, date):
        raise NotImplementedError

    @abc.abstractmethod
    def place_order(self, account_id, order):
        raise NotImplementedError

    @abc.abstractmethod
    def cancel_order(self, account_id, order):
        raise NotImplementedError

    @abc.abstractmethod
    def get_open_orders(self):
        raise NotImplementedError

    @abc.abstractmethod
    def subscribe(self, symbol_list):
        raise NotImplementedError
