import abc
import logging

from six import with_metaclass

class BaseExtension(with_metaclass(abc.ABCMeta)):

    @abc.abstractmethod
    def create(self, _context):
        raise NotImplementedError

