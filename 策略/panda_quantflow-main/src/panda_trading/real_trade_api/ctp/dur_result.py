class DurResult(object):

    spi_dict = dict()

    def __init__(self):
        pass

    @classmethod
    def save_spi(cls, account, spi):
        cls.spi_dict[account] = spi
