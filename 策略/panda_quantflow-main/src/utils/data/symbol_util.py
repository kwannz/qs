from datetime import datetime
import re


class SymbolUtil:

    @staticmethod
    def czc_symbol_chagne(symbol, trade_date):
        """
        郑商所合约根据交易日期换成相应完整合约号（如：AP001.CZC在2010年换成AP1001.CZC）
        :param symbol:
        :param trade_date:
        :return:
        """
        num = str(trade_date)[2:3]
        new = re.sub(r"\D", "", symbol)
        if len(new) == 3:
            new = num + new
            symbol = re.sub(r"\d+", new, symbol)
        return symbol.upper()

    @staticmethod
    def get_symbol(emcode):
        """
        将合约进行转换（针对郑商所合约号需要去掉年份）
        :param emcode:
        :return:
        """
        exchange = emcode.split('.')[1]
        if exchange == 'CZC':
            new = re.sub(r"\D", "", emcode)
            new = new[1:]
            symbol = re.sub(r"\d+", new, emcode).upper()
        else:
            symbol = emcode.upper()
        return symbol

    @staticmethod
    def symbol_to_ctp_code(symbol):
        """
        平台合约转ctp合约
        :param symbol:
        :return:
        """
        if len(symbol.split('.')) < 2:
            # 组合合约
            return symbol
        code = symbol.split('.')[0]
        market = symbol.split('.')[1]
        if market == 'SHF':
            ctp_code = code.lower()
        elif market == 'CZC':
            new = re.sub(r"\D", "", code)
            new = new[1:]
            ctp_code = re.sub(r"\d+", new, code).upper()
        elif market == 'DCE':
            ctp_code = code.lower()
        elif market == 'CFE':
            ctp_code = code
        elif market == 'INE':
            ctp_code = code.lower()
        elif market == 'GFE':
            ctp_code = code.lower()
        else:
            ctp_code = code
        return ctp_code

    @staticmethod
    def code_to_ctp_code(code, market):
        """
        平台短合约转ctp合约
        :param code:
        :param market:
        :return:
        """
        ctp_code = ''
        if market == 'SHFE':
            ctp_code = code.lower()
        elif market == 'CZCE':
            new = re.sub(r"\D", "", code)
            new = new[1:]
            ctp_code = re.sub(r"\d+", new, code).upper()
        elif market == 'DCE':
            ctp_code = code.lower()
        elif market == 'CFFEX':
            ctp_code = code
        elif market == 'INE':
            ctp_code = code.lower()
        elif market == 'GFEX':
            ctp_code = code.lower()
        else:
            ctp_code = code
        return ctp_code

    @staticmethod
    def ctp_code_to_code(ctp_code, market):
        """
        ctp合约装平台合约
        :param ctp_code:
        :param market:
        :return:
        """
        if '&' in ctp_code:
            return ctp_code
        code = ''
        if market == 'SHFE':
            code = ctp_code.upper() + '.SHF'
        elif market == 'CZCE':
            today = datetime.now().strftime('%Y%m%d')
            num = str(today)[2:3]
            year = str(today)[3:4]
            new = re.sub(r"\D", "", ctp_code)
            if len(new) == 3:
                data_year = new[0:1]
                if int(data_year) >= int(year):
                    new = num + new
                    code = re.sub(r"\d+", new, ctp_code).upper() + '.CZC'
                else:
                    if int(num) == 9:
                        num = str(0)
                    else:
                        num = str(int(num) + 1)
                    new = str(num) + new
                    code = re.sub(r"\d+", new, ctp_code).upper() + '.CZC'
        elif market == 'DCE':
            code = ctp_code.upper() + '.DCE'
        elif market == 'CFFEX':
            code = ctp_code + '.CFE'
        elif market == 'INE':
            code = ctp_code.upper() + '.INE'
        elif market == 'GFEX':
            code = ctp_code.upper() + '.GFE'
        else:
            code = ctp_code
        return code

    @staticmethod
    def quant_ts_code_to_code(quant_ts_code):
        """
        quantos行情合约转平台合约
        :param quant_ts_code:
        :return:
        """
        exchange = quant_ts_code.split('.')[1]
        if exchange == 'CZC':
            today = datetime.now().strftime('%Y%m%d')
            num = str(today)[2:3]
            year = str(today)[3:4]
            new = re.sub(r"\D", "", quant_ts_code)
            if len(new) == 3:
                # 需要判断09/19等换年情况
                data_year = new[0:1]
                if int(data_year) >= int(year):
                    new = num + new
                    code = re.sub(r"\d+", new, quant_ts_code).upper()
                else:
                    if int(num) == 9:
                        num = str(0)
                    else:
                        num = str(int(num) + 1)
                    new = str(num) + new
                    code = re.sub(r"\d+", new, quant_ts_code).upper()
            else:
                code = quant_ts_code.upper()
        else:
            code = quant_ts_code.upper()

        return code

    @staticmethod
    def xtp_market_to_market(market):
        if market == 1:
            return 'SZ'
        elif market == 2:
            return 'SH'
        else:
            return ''

    @staticmethod
    def tora_market_to_market(market):
        if market == '2':
            return 'SZ'
        elif market == '1':
            return 'SH'
        else:
            return ''


if __name__ == '__main__':
    print( SymbolUtil.quant_ts_code_to_code('LC2412.GFE'))
