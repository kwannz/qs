# 框架基本方法

## 基础方法说明
> 该策略为事件驱动性策略，需要实现框架中约定的事件回调方法，实现后回测、仿真、实盘通用。

> 策略头部需要默认引用内置API，代码为：from panda_backtest.api.api import *，后文不再重复赘述。

> 接下来具体介绍框架各个事件回调方法，必选代表必须在策略中实现。


### 策略初始化（必选）

>函数：initialize
>描述：策略初始化,主要用于初始化策略上下文中的变量，只在策略启动时运行一次
*****
> 代码
``` python
def initialize(context):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| context | Context对象 | 策略上下文对象 |
*****
> 例子
``` python
def init(context):
    # 上下文中保存run_flag变量，可以在运行函数handle_bar中通过context对象调取
    context.run_flag = true
```

### 策略bar运行（必选）


>函数：handle_data
>描述：策略bar触发运行函数,日线回测为每日调用一次，分钟则为每个交易分钟时间调用一次
>说明：当有基金交易时，分为普通交易和所有时间交易，分钟运行时间参考下表：
>
| 策略类型 | 运行时间 |
| --- | --- |
| 股票 | 9:30 ~ 15:00 |
| 期货 | 9:00 ~ 15:00 |
| 基金（普通） | 9:30 ~ 15:00 |
| 基金（所有时间） | 00:00 ~ 23:59 |
| 混合（所有时间） | （上个交易日）15:31 ~ 15:30 |
| 混合（有期货） | （上个交易日）20:30 ~ 15：00 |
| 混合（无期货） | 9:30 ~ 15:00 |
*****
> 代码
``` python
def handle_data(context, bar):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| context | Context对象 | 策略上下文对象 |
| bar | Bar对象 | bar行情对象 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 打印平安银行当前回测k线收盘价
    SRLogger.info(bar['000001.SZ'].close)
    # 打印黄金2002合约当前回测k线收盘价
    SRLogger.info(bar['AU2002.SHF'].close)
    # 股票账号以市价买入2000股平安银行
    order_shares('8888', '000001.SZ', 2000, style=MarketOrderStyle)
    # 期货账号以市价开仓买入黄金2002合约1手
    buy_open("5588","AU2002.SHF",1, style=MarketOrderStyle)
```

### 策略开盘前


>函数：before_trading
>描述：策略开盘前运行函数
>注意：该函数只在交易日调用，分钟回测调用时间参考下表
>
| 策略类型 | 运行时间 |
| --- | --- |
| 股票 | 8:30 |
| 期货 | 20:30 |
| 基金（普通） | 8:30 |
| 基金（所有时间） | 15:31 |
| 混合（所有时间） | 15:31 |
| 混合（有期货） | 20:30 |
| 混合（无期货） | 8:30 |
*****
> 代码
``` python
def before_trading(context):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| context | Context对象 | 策略上下文对象 |
*****
> 例子
``` python
def before_trading(context):
    # 开盘前打印
    total_value = context.future_account_dict['5588'].total_value
    SRLogger.info(total_value)
```


### 策略收盘后


>函数：after_trading
>描述：策略收盘后运行函数
>注意：该函数只在交易日调用，调用时间为15:30
*****
> 代码
``` python
def after_trading(context):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| context | Context对象 | 策略上下文对象 |
*****
> 例子
``` python
def after_trading(context):
    # 在收盘后打印当前期货账号的总权益
    total_value = context.future_account_dict['5588'].total_value
    SRLogger.info(total_value)
```

### 股票报单回报
>函数：on_stock_trade_rtn
>描述：当有报单委托成交后触发
*****
> 代码
``` python
def on_stock_trade_rtn(context, order):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| context | Context对象 | 策略上下文对象 |
| order | Order对象 | 订单信息 |
*****
> 例子
``` python
def on_stock_trade_rtn(context, order):
    # 打印订单标的
    order_book_id = order.order_book_id
    SRLogger.info(order_id)
    # 打印订单手数
    quantity = order.quantity
    SRLogger.info(quantity)
```
### 股票撤单回报
>函数：stock_order_cancel
>描述：当有报单委托被撤单后触发
*****
> 代码
``` python
def stock_order_cancel(context, order):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| context | Context对象 | 策略上下文对象 |
| order | Order对象 | 订单信息 |
*****
> 例子
``` python
def stock_order_cancel(context, order):
    # 打印订单标的
    order_book_id = order.order_book_id
    SRLogger.info(order_id)
    # 打印订单手数
    quantity = order.quantity
    SRLogger.info(quantity)
```

### 期货报单回报
>函数：on_future_trade_rtn
>描述：当有报单委托成交后触发
*****
> 代码
``` python
def on_future_trade_rtn(context, order):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| context | Context对象 | 策略上下文对象 |
| order | Order对象 | 订单信息 |
*****
> 例子
``` python
def on_future_trade_rtn(context, order):
    # 打印订单标的
    order_book_id = order.order_book_id
    SRLogger.info(order_id)
    # 打印订单手数
    quantity = order.quantity
    SRLogger.info(quantity)
```

### 期货撤单回报
>函数：future_order_cancel
>描述：当有报单委托被撤单后触发
*****
> 代码
``` python
def future_order_cancel(context, order):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| context | Context对象 | 策略上下文对象 |
| order | Order对象 | 订单信息 |
*****
> 例子
``` python
def future_order_cancel(context, order):
    # 打印订单标的
    order_book_id = order.order_book_id
    SRLogger.info(order_id)
    # 打印订单手数
    quantity = order.quantity
    SRLogger.info(quantity)
```

### 事件拓展
> 系统支持自定义事件，详细参考开源代码src/panda_backtest/backtest_common/system/event/event.py

## 基础对象说明

### context对象

>对象：**context**
>描述：全局上下文对象，可在基础函数中传递，同时会内置数据对象
*****
> 代码
``` python
context.变量
```
*****
> 内置变量
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| now | str | 当前日期（yyyMMdd） |
| portfolio_dict | dict | 收益信息字典（key为account，value为Portfolio对象）|
| stock_account_dict | dict | 股票账户信息字典（key为account，value为StockAccount对象）|
| future_account_dict | dict | 期货账户信息字典（key为account，value为FutureAccount对象）|
| df_factor | DataFrame | 因子数据框，由因子构建节点生成的因子值数据 |
*****
> 例子
``` python
stock_account = context.stock_account_dict['8888']
```


### Bar对象

>对象：**bar**
>描述：当前bar行情对象
*****
> 代码
``` python
bar[合约]
```
*****
> 内置变量
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| symbol | str | 合约 |
| open | double | 开盘价 |
| high | double | 最高价 |
| low | double | 最低价 |
| close | double | 收盘价 |
| settle | double | 结算价 |
| last | double | 最新价 |
| volume | long | 成交量 |
| oi | long | 持仓量 |
| turnover | double | 成交金额 |
*****
> 重要: bar 不是 python 内置字典类型, 需要以 bar['行情对象'] 方式来使用. 无法使用 bar.keys(), bar.items(), bar.values(), bar.__contains__(key), bar.__iter__(),bar.__len__() 等方法, 也不能进行遍历和使用in方法判断.
> 例子
``` python
# 获取平安银行当前bar收盘价
close = bar['000001.SZ'].close
```


### df_factor对象

>对象：**context.df_factor**
>描述：因子数据框对象，由因子构建节点生成的因子值数据，用于策略中的因子分析
*****
> 代码
``` python
context.df_factor
```
*****
> 数据结构
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| date（index） | str | 日期索引（格式：yyyyMMdd），作为 DataFrame 复合索引的第一级 |
| symbol（index） | str | 股票/期货合约代码（如：000001.SZ，AG2002.SHF），作为 DataFrame 复合索引的第二级 |
| factor_value | float | 因子值，用于排序和选股的数值 |
| [其他列] | any | 根据具体因子构建节点的输出可能包含其他列 |

> **索引结构说明**
> 该 DataFrame 使用 `date` 和 `symbol` 作为复合索引（MultiIndex）
> 在使用时，如果需要将复合索引转换为普通列进行查询，可以使用 `reset_index()` 方法。

> 例子
``` python
def initialize(context):
    # 预处理因子数据
    # 注意：date 是 DataFrame 的 index，需要 reset_index() 将其转换为列以便查询
    context.df_factor = context.df_factor.reset_index()
    context.df_factor['factor_value'] = pd.to_numeric(
        context.df_factor
        .groupby('symbol')[context.df_factor.columns[2]]
        .shift(1),
        errors='coerce'
    )

def handle_data(context, bar):
    today = context.now
    # 获取今日因子值并按值排序
    # 注意：date 是 DataFrame 的 index，需要 reset_index() 将其转换为列以便查询
    context.df_factor = context.df_factor.reset_index()
    df_today = context.df_factor[context.df_factor["date"] == today]
    df_today_sorted = df_today.sort_values('factor_value', ascending=False)
    # 选择前N只股票
    buy_list = df_today_sorted.head(5)['symbol'].tolist()
```


### Order对象

>对象：**Order**
>描述：下单返回订单对象
*****
> 代码
``` python
order = order_shares('8888','000001.SZ', 100 )
```
*****
> 内置变量
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order_id | str | 订单唯一标识 |
| order_book_id | str | 下单合约 |
| side | int | 买卖方向（1：买 2：卖） |
| effect | int | 开平方向（0：开，1：平） |
| price | double | 订单价格，只有在订单类型为'限价单'的时候才有意义 |
| quantity | int | 下单数量 |
| filled_quantity | int | 订单已成交数量 |
| unfilled_quantity | int | 订单剩余数量 |
| status | int | 订单状态（1：未成交，2：已成交，3：已撤，-1：拒单） |
| message | str | 订单信息 |
*****
> 例子
``` python
# 获取订单id
order = order_shares('8888','000001.SZ', 100 )
order_id = order.order_id
```


### StockAccount对象

>对象：**StockAccount**
>描述：股票账号信息实体对象
*****
> 代码
``` python
context.stock_account_dict['8888']
```
*****
> 内置变量
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| total_value | double | 总权益 |
| cash | double | 可用资金 |
| frozen_cash | double | 冻结资金 |
| market_value | double | 持仓市值 |
| positions | dict | 持仓字典（key为合约，value为StockPositions对象） |
*****
> 例子
``` python
# 获取股票账号总权益
total_value = context.stock_account_dict['8888'].total_value
```


### StockPosition对象

>对象：**StockPositions**
>描述：股票持仓对象
*****
> 代码
``` python
context.stock_account_dict['8888'].positions['000001.SZ']
```
*****
> 内置变量
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order_book_id | str | 合约 |
| quantity | int | 持仓数量 |
| sellable | int | 可卖数量 |
| market_value | double | 持仓市值 |
| avg_price | double | 持仓均价 |
| pnl | double | 持仓盈亏 |
*****
> 例子
``` python
# 获取股票账号平安银行持仓数量
quantity = context.stock_account_dict['8888'].positions['000001.SZ'].quantity
```


### FutureAccount对象

>对象：**FutureAccount**
>描述：期货账号信息实体对象
*****
> 代码
``` python
context.future_account_dict['5588']
```
*****
> 内置变量
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| total_value | double | 总权益 |
| cash | double | 可用资金 |
| frozen_cash | double | 冻结资金 |
| holding_pnl | double | 持仓盈亏 |
| realized_pnl | double | 平仓盈亏 |
| margin | double | 保证金 |
| transaction_cost | double | 手续费 |
| positions | dict | 持仓字典（key为合约，value为FuturePositions对象） |
*****
> 例子
``` python
# 获取期货账号总权益
total_value = context.future_account_dict['5588'].total_value
```


### FuturePosition对象

>对象：**FuturePositions**
>描述：期货持仓对象
*****
> 代码
``` python
context.stock_account_dict['8888'].positions['000001.SZ']
```
*****
> 内置变量
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order_book_id | str | 合约 |
| buy_quantity | int | 多头持仓 |
| buy_today_quantity | int | 多头今日持仓 |
| closable_buy_quantity | int | 多头可平持仓 |
| buy_margin | double | 多头保证金 |
| buy_pnl | double | 多头累计收益 |
| buy_avg_open_price | double | 多头开仓均价 |
| buy_avg_holding_price | double | 多头持仓均价 |
| buy_transaction_cost | double | 多头手续费 |
| buy_pnl | double | 多头累计收益 |
| sell_quantity | int | 空头持仓 |
| sell_today_quantity | int | 空头今日持仓 |
| closable_sell_quantity | int | 空头可平持仓 |
| sell_margin | double | 空头保证金 |
| sell_pnl | double | 空头累计收益 |
| sell_avg_open_price | double | 空头开仓均价 |
| sell_avg_holding_price | double | 空头持仓均价 |
| sell_transaction_cost | double | 空头手续费 |
| sell_pnl | double | 空头累计收益 |
*****
> 例子
``` python
# 获取期货账号AG2002合约多头仓位
buy_quantity = context.future_account_dict['5588'].positions['AG2002.SHF'].buy_quantity
```


### FundAccount对象


>对象：**FundAccount**
>描述：基金账号信息实体对象
*****
> 代码
``` python
context.fund_account_dict['2233']
```
*****
> 内置变量
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| total_value | double | 总权益 |
| cash | double | 可用资金 |
| frozen_cash | double | 冻结资金 |
| market_value | double | 持仓市值 |
| positions | dict | 持仓字典（key为合约，value为StockPositions对象） |
*****
> 例子
``` python
# 获取股票账号总权益
total_value = context.fund_account_dict['2233'].total_value
```

### FundPosition对象


>对象：**FundPositions**
>描述：基金持仓对象
*****
> 代码
``` python
context.fund_account_dict['2233'].positions['000311.OF']
```
*****
> 内置变量
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order_book_id | str | 合约 |
| quantity | int | 持仓数量 |
| sellable | int | 可卖数量 |
| market_value | double | 持仓市值 |
| avg_price | double | 持仓均价 |
| pnl | double | 持仓盈亏 |
*****
> 例子
``` python
# 获取景顺沪深300持仓数量
quantity = context.fund_account['2233'].positions['000311.OF'].quantity
```


# 交易函数
## 股票交易
### 指定股数下单

>函数：order_shares
>描述：指定股数进行股票交易
*****
> 代码
``` python
def order_shares(account, symbol, amount, style):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| account | str | 股票账号 |
| symbol | str | 股票合约 |
| amount | int | 股数（正数代表买入，负数代表卖出） |
| style | enum | 订单类型, MarketOrderStyle=市价单,LimitOrderStyle=限价单 |

> 输出参数

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order | Order对象 | 订单对象 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 按照市价最新价买入100股平安银行
    order_shares('8888','000001.SZ', 100)
    # 按照市价最新价卖出100股平安银行
    order_shares('8888','000001.SZ', -100)
```
或者
```python
def handle_data(context, bar):
    # 按照12.89价格买入100股平安银行
    order_shares('8888','000001.SZ', 100, style=LimitOrderStyle(12.89))
```

### 按照目标持仓下单

>函数：target_stock_group_order
>描述：按照目标持仓下单，在1分钟内，以最小代价，将当前持仓改为目标持仓
*****
> 代码
``` python
def target_stock_group_order(account, symbol_dict):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| account | str | 股票账号 |
| symbol | dict | 股票合约和股数 （{"000001.SZ":100}）|
> 输出参数

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order | Order对象 | 订单对象 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 平掉当前持仓，买入中国平安1手
    target_stock_group_order('8888',{'000001.SZ':100})
```


### 撤单

>函数：cancel_order
>描述：股票撤单，一般只用于限价单挂单，市价单为即成即撤无法撤单
*****
> 代码
``` python
def cancel_order(account, order_id):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| account | str | 股票账号 |
| order_id | str | 订单id |
> 输出参数

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| result | bool | 是否撤单成功 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 按照市价买入100股平安银行
    order_list = order_shares('8888','000001.SZ', 100)
    # 对订单进行撤单
    for order in order_list:
        cancel_order('8888',order.order_id)
```

## 期货交易
### 买入开仓

>函数：buy_open
>描述：期货买入开仓
*****
> 代码
``` python
def buy_open(account, symbol, amount, style):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| account | str | 期货账号 |
| symbol | str | 期货合约 |
| amount | int | 手数 |
| style | enum | 订单类型, MarketOrderStyle=市价单,LimitOrderStyle=限价单 |
> 输出参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order | Order对象 | 订单对象 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 按照市价最新价开仓买入1手AG2002
    buy_open('5588','AG2002.SHF', 1)
```
或者
``` python
def handle_data(context, bar):
    # 按照4280价格开仓买入1手AG2002
    buy_open('5588','AG2002.SHF', 1, style=LimitOrderStyle(4280))
```


### 卖出开仓

>函数：sell_open
>描述：期货卖出开仓
*****
> 代码
``` python
def sell_open(account, symbol, amount, style):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| account | str | 期货账号 |
| symbol | str | 期货合约 |
| amount | int | 手数 |
| style | enum | 订单类型, MarketOrderStyle=市价单,LimitOrderStyle=限价单 |
> 输出参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order | Order对象 | 订单对象 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 按照市价最新价开仓卖出1手AG2002
    sell_open('5588','AG2002.SHF', 1)
```
或者
``` python
def handle_data(context, bar):
    # 按照4280价格开仓卖出1手AG2002
    sell_open('5588','AG2002.SHF', 1, style=LimitOrderStyle(4280))
```

### 买入平仓

>函数：buy_close
>描述：期货买入平仓，即平空头仓位
*****
> 代码
``` python
def buy_close(account, symbol, amount, style):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| account | str | 期货账号 |
| symbol | str | 期货合约 |
| amount | int | 手数 |
| style | enum | 订单类型, MarketOrderStyle=市价单,LimitOrderStyle=限价单 |
> 输出参数

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order | Order对象 | 订单对象 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 按照市价最新价平仓1手AG2002空头
    buy_close('5588','AG2002.SHF', 1)
```
或者
``` python
def handle_data(context, bar):
    # 按照4280价格平仓1手AG2002空头
    buy_close('5588','AG2002.SHF', 1, style=LimitOrderStyle(4280))
```

#### 卖出平仓

>函数：sell_close
>描述：期货卖出平仓，即平多头仓位
*****
> 代码
``` python
def sell_close(account, symbol, amount, style):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| account | str | 期货账号 |
| symbol | str | 期货合约 |
| amount | int | 手数 |
| style | enum | 订单类型, MarketOrderStyle=市价单,LimitOrderStyle=限价单 |
> 输出参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order | Order对象 | 订单对象 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 按照市价最新价平仓1手AG2002多头
    sell_close('5588','AG2002.SHF', 1)
```
或者
``` python
def handle_data(context, bar):
    # 按照4280价格平仓1手AG2002多头
    sell_close('5588','AG2002.SHF', 1, style=LimitOrderStyle(4280))
```

### 按照目标持仓下单

>函数：target_future_group_order
>描述：按照目标持仓下单，在1分钟内，以最小代价，将当前持仓改为目标持仓
*****
> 代码
``` python
def target_future_group_order(account, long_symbol_dict, short_symbol_dict):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| account | str | 股票账号 |
| long_symbol_dict | dict | 多头期货合约和股数 （{"AG2509.SHF":100}）|
| short_symbol_dict | dict | 空头期货合约和股数 （{"AG2509.SHF":100}）|
> 输出参数

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order | Order对象 | 订单对象 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 平掉当前持仓，建立多头AG2505.SHF，空头A2505.DCE
    long_dict = {"AG2505.SHF": 1}
    short_dict = {"A2505.DCE": 1}
    target_future_group_order('8888', long_dict, short_dict)
```

### 期货撤单

>函数：cancel_future_order
>描述：期货撤单，一般只用于限价单挂单，市价单为即成即撤无法撤单
*****
> 代码
``` python
def cancel_future_order(account, order_id):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| account | str | 股票账号 |
| order_id | str | 订单id |
> 输出参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| result | bool | 是否撤单成功 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 按照市价最新价开仓买入1手AG2002
    order = buy_open('8888','000001.SZ', 100, style=LimitOrderStyle(4280))
    # 对订单进行撤单
    cancel_order('5588',order.order_id)
```

## 基金交易

### 申购基金

>函数：purchase
>描述：按照指定金额申购基金
*****
> 代码
``` python
def purchase(account, symbol, amount, fund_cover_old=False):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| account | str | 基金账号 |
| symbol | str | 基金合约 |
| amount | int | 金额 |
| fund_cover_old | bool | 是否覆盖老订单 |
> 输出参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order | Order对象 | 订单对象 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 申购70000块景顺沪深300
    purchase('2233', '000311.OF', 700000)
```

### 赎回基金

>函数：redeem
>描述：按照指定份额赎回基金
*****
> 代码
``` python
def redeem(account, symbol, quantity, fund_cover_old=False):
```
*****
> 参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| account | str | 基金账号 |
| symbol | str | 基金合约 |
| quantity | int | 份额 |
| fund_cover_old | bool | 是否覆盖老订单 |
> 输出参数
> 
| 字段 | 类型 | 描述 |
| --- | --- | --- |
| order | Order对象 | 订单对象 |
*****
> 例子
``` python
def handle_data(context, bar):
    # 赎回100份景顺沪深300
    redeem('2233', '000311.OF', 100)
```

# 标的

| | 股票/场内基金 |
| --- | --- |
| 深圳证券交易所 | SZ |
| 上海证券交易所 | SH |

| | 期货 |
| --- | --- |
| 上海期货交易所 | SHF |
| 大连商品交易所 | DCE |
| 郑州商品交易所 | CZC |
| 上海能源交易所 | INE |
| 中国金融期货交易所 | CFE |

| 场外基金 |
| --- |
| OF |

# 案例
## 根据期货因子下单，因子格式为dataframe，列为symbol、factor_name、value
``` python
from panda_backtest.api.api import *
from panda_backtest.api.future_api import *
import pandas as pd
import numpy as np
import copy
import datetime
import re
import pickle
import sys

def initialize(context):
    # 策略参数设置，可以根据实际情况进行修改
    context.s_top_n = 5               # 每次买入的前N只标的
    context.s_rb_period = 5            # 调仓周期（单位：天）

    # 预处理因子数据
    # 注意：date 是 DataFrame 的 index，需要 reset_index() 将其转换为列
    context.df_factor = context.df_factor.reset_index()
    context.df_factor['factor_value'] = pd.to_numeric(
        context.df_factor
        .groupby('symbol')[context.df_factor.columns[2]]
        .shift(1),
        errors='coerce'
    )

    SRLogger.info("策略初始化完成")
    context.account = '5588'


def handle_data(context, bar):
    if int(context.now) % context.s_rb_period != 0:
        return  # 非调仓日不执行任何操作

    SRLogger.info(f"调仓日：{context.now}")
    today = context.now

    # 获取今日因子值并按值排序
    df_today = context.df_factor[context.df_factor["date"] == today]
    df_today_sorted = df_today.sort_values('factor_value', ascending=False)
    buy_list = df_today_sorted.head(context.s_top_n)['symbol'].tolist()

    # 获取行情数据
    quotation_df = future_api_quotation(symbol_list=buy_list, start_date=today, end_date=today, period="1d")
    # 获取主力合约
    quotation_df['symbol'] = quotation_df['dominant_id'] + '.' + quotation_df['exchange']

    per_close = quotation_df.set_index('symbol')['close'].to_dict()
    symbols = list(per_close.keys())

    # 获取合约乘数
    contract_mul = future_api_symbol_contractmul(symbols).set_index('symbol')['contractmul'].to_dict()

    total_value = context.future_account_dict[context.account].total_value

    # 构建下单指令
    orders = {}
    for symbol in symbols:
        if symbol not in contract_mul or symbol not in per_close:
            SRLogger.warning(f"缺失数据: {symbol}")
            continue

        hands = total_value / (contract_mul[symbol] * per_close[symbol] * context.s_top_n)
        hands = np.floor(np.abs(hands))
        orders[symbol] = hands

        SRLogger.info(f"{symbol}: 合约乘数={contract_mul[symbol]}, 收盘价={per_close[symbol]}, 下单手数={hands}")

    SRLogger.info(pd.DataFrame(list(orders.items()), columns=['symbol', 'order_hands']))
    target_future_group_order(context.account, orders, {})

```
