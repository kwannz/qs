

# 订单信息
STOCK_ORDER_FAILED_MESSAGE = '股票报单失败，合约：【%s】,股数：【%s】股, 账号：【%s】,%s, %s, 订单id:%s, 信息：%s'
FUTURE_ORDER_FAILED_MESSAGE = '期货报单失败，合约：【%s】,手数：【%s】手, 账号：【%s】,%s, %s, 订单id:%s, 信息：%s'
FUND_ORDER_FAILED_MESSAGE = '基金报单失败，合约：【%s】,手数：【%s】手, 账号：【%s】,%s, %s, 订单id:%s, 信息：%s'
ORDER_CANCEL_FAILED_MESSAGE = '撤单失败， 账号：【%s】，订单id: 【%s】, 信息：%s'
ORDER_CASH_NOT_ENOUGH = '资金不足，下单资金：%s，当前可用资金: %s'
ORDER_POSITION_NOT_ENOUGH = '仓位不足，当前仓位: %s'
SYMBOL_LIMIT_HIGH = '合约涨停，无法买入'
SYMBOL_LIMIT_LOW = '合约跌停，无法卖出'
SYMBOL_LIMIT_NOT_EXIST = '无法获取合约涨跌停价格'
ORDER_PRICE_TOO_HIGH = '合约报价：%s,超过涨停价：%s'
ORDER_PRICE_TOO_LOW = '合约报价：%s,超过跌停价：%s'
STOCK_HAD_NOT_QUOTATION = '合约不存在行情数据'
STOCK_SU_SP = '合约停牌'
SYMBOL_NO_QUOTATION = '合约不存在行情'
SYMBOL_PRICE_NOT_RIGHT = '报单价格有误'
FUTURE_NOT_LIMIT_DATA = '无法获取涨跌停数据'
STOCK_GEM_QUANTITY_NOT_RIGHT = '创业板下单数量错误，最低200股，当前下单股数：%s股'
STOCK_HAD_NO_INFO = '当前未收录该股票信息'
STOCK_QUANTITY_NOT_RIGHT = '下单数量错误，最低100股，当前下单股数：%s股'
STOCK_NO_VOLUME = '合约无市场成交(停牌或冷门合约)，无法进行成交'
STOCK_VOLUME_NOT_ENOUGH = '下单数量超过当前市场成交总量，成交总量：%s'
FUND_REDEEM_QUANTITY_NOT_RIGHT = '赎回份额应大于0，当前赎回份额：%s份'
FUND_PURCHASE_QUANTITY_NOT_RIGHT = '申购金额应大于0，当前申购金额：%s元'
FUND_HAD_NO_INFO = '当前未收录该基金信息'
CANCEL_ORDER_FAILED_MESSAGE = '撤单失败，账号：【%s】, 订单id:%s, 信息：%s'
SYMBOL_NOT_TRADE_IN_THIS_TIME = '合约未在交易时间'
SYMBOL_CAN_NOT_CROSS = '已撤单，订单合约报价撮合失败'

# 分红信息
STOCK_DIVIDEND_INFO = '股票进行分红，账号：【%s】，合约：【%s】,现金：【%s】，送股：【%s】'
FUND_DIVIDEND_INFO = '基金进行分红，账号：【%s】，合约：【%s】,分红方式：现金分红, 现金：【%s】'

# 基金拆分
FUND_SPLIT_INFO = '基金进行份额拆分，账号：【%s】，合约：【%s】,拆分前份额：【%s】,拆分后份额：【%s】，拆分前均价：【%s】，拆分后均价：【%s】'
ETF_SPLIT_INFO = 'ETF进行份额拆分，账号：【%s】，合约：【%s】,拆分前份额：【%s】,拆分后份额：【%s】，拆分前均价：【%s】，拆分后均价：【%s】'

# 基金到账
FUND_ARRIVE_INFO = '基金赎回到账，账号：【%s】，合约：【%s】, 到账金额：【%s】'

# 实盘相关错误
CTP_INSERT_ORDER_ERROR = 'ctp下单错误，错误码：{}'
FUTURE_ACCOUNT_NOT_INIT = '期货账号未初始化'
FUTURE_ORDER_ID_NOT_INIT = '期货报单序号未初始化'
TORA_ORDER_ID_NOT_INIT = 'tora系统报单序号未初始化'
TORA_INSERT_ORDER_ERROR = 'tora下单错误，错误码：{}'
TORA_ACCOUNT_NOT_INIT = 'tora账号未初始化'

# 微信消息推送
FUTURE_ORDER_WX_MESSAGE = '当前策略触发期货账号调仓信号，请注意查看，产品：{}，策略名称：{}，账号：{}'
