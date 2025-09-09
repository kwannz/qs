import abc
import logging

from six import with_metaclass

class BaseEventProcess(with_metaclass(abc.ABCMeta)):

    @abc.abstractmethod
    def event_factory(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_backtest_params(
            self,handle_message):
        raise NotImplementedError

