#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午9:14
# @Author : wlb
# @File   : index_calculate.py
# @desc   : 计算策略指标的api
import numpy as np
import logging

import pandas as pd
import sys
import math

class IndexCalculate(object):

    @staticmethod
    def new_beta(_portfolio, _benchmark):
        if len(_portfolio) < 2:
            _beta = np.nan
            return _beta

        print('_portfolio===>' + str(len(_portfolio)))
        print('_benchmark===>' + str(len(_benchmark)))
        cov = np.cov(np.vstack([
            _portfolio,
            _benchmark
        ]), ddof=1)
        _beta = cov[0][1] / cov[1][1]
        return _beta

    @staticmethod
    def beta(date_line, return_line, indexreturn_line):
        """
        :param date_line: 日期序列
        :param return_line: 账户日收益率序列
        :param indexreturn_line: 指数的收益率序列
        :return: 输出beta值
        """
        if len(return_line) < 2:
            print('beta: %f' % np.nan)
            return np.nan

        df = pd.DataFrame({'date': date_line,
                           'rtn': return_line,
                           'benchmark_rtn': indexreturn_line})
        # 账户收益和基准收益的协方差除以基准收益的方差
        b = df['rtn'].cov(df['benchmark_rtn']) / df['benchmark_rtn'].var()
        print('beta: %f' % b)
        return b

    @staticmethod
    def annual_return(end_account_rate, date_num):
        if date_num == 0:
            return 0
        if 1 + end_account_rate < 0:
            annual_return_value = -1 * math.pow(
                abs(end_account_rate) - 1, 250 / date_num) - 1
        else:
            annual_return_value = math.pow(
                1 + end_account_rate, 1 / (date_num / 250)) - 1
        print('策略年化收益：%s' % str(annual_return_value))
        return annual_return_value

    @staticmethod
    def standard_symbol_return(end_standard_symbol_rate, date_num):
        standard_symbol_return_value = pow(
            1 + end_standard_symbol_rate, 1 / (date_num / 250)) - 1
        print('基准年化收益：%s' % str(standard_symbol_return_value))
        return standard_symbol_return_value

    @staticmethod
    def volatility(account_daily_income_rate):
        if len(account_daily_income_rate) < 2:
            volatility = 0.
            annual_volatility = 0.
        else:
            volatility = np.array(account_daily_income_rate).std(ddof=1)
            annual_volatility = volatility * (250 ** 0.5)
        print('波动收益率：' + str(volatility))
        print('年化波动收益率：' + str(annual_volatility))
        return annual_volatility

    @staticmethod
    def max_drawdown(date_line, capital_line):
        """
        :param date_line: 日期序列
        :param capital_line: 账户价值序列
        :return: 输出最大回撤及开始日期和结束日期
        """
        if len(date_line) == 0:
            return 0
        # 将数据序列合并为一个dataframe并按日期排序
        df = pd.DataFrame({'date': date_line, 'capital': capital_line})
        df.sort_values(by='date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # df['max2here'] = pd.expanding_max(df['capital'])  # 计算当日之前的账户最大价值
        df['max2here'] = df['capital'].expanding(min_periods=1).max()  # 计算当日之前的账户最大价值
        df['dd2here'] = df['capital'] / df['max2here'] - 1  # 计算当日的回撤

        # 计算最大回撤和结束时间
        temp = df.sort_values(by='dd2here').iloc[0][['date', 'dd2here']]
        max_dd = temp['dd2here']
        end_date = temp['date']

        # 计算开始时间
        df = df[df['date'] <= end_date]
        start_date = df.sort_values(
            by='capital', ascending=False).iloc[0]['date']

        print('最大回撤为：%f, 开始日期：%s, 结束日期：%s' % (max_dd, start_date, end_date))
        return max_dd

    @staticmethod
    def sharpe_ratio(
            strategy_year_income_rate,
            risk_free_rate,
            volatility):
        """
        计算夏普比率
        :param strategy_year_income_rate: 策略年化收益率
        :param risk_free_rate: 无风险收益率
        :param volatility: 策略波动率
        :return:
        """
        if volatility == 0.0:
            return np.nan
        sharpe = (strategy_year_income_rate - risk_free_rate) / volatility
        print('夏普比率：%s:' % str(sharpe))
        return sharpe

    @staticmethod
    def alpha(
            strategy_year_income_rate,
            risk_free_rate,
            standard_income_rate,
            beta):
        """
        计算阿尔法指标
        :param strategy_year_income_rate: 策略年化收益率
        :param risk_free_rate: 无风险组合收益率
        :param standard_income_rate: 基准指标年化收益率
        :param beta: 贝塔指标
        :return:
        """
        al = strategy_year_income_rate - risk_free_rate - \
             beta * (standard_income_rate - risk_free_rate)
        print('alpha比率%f:' % al)
        return al

    # 计算信息比率函数
    @staticmethod
    def info_ratio(
            account_daily_income_rate,
            standard_symbol_income_rate,
            annual_te):

        if len(account_daily_income_rate) < 2:
            information_ratio = None
            return information_ratio

        if annual_te == 0:
            information_ratio = None
            return information_ratio

        active_return = np.array(account_daily_income_rate) - \
                        np.array(standard_symbol_income_rate)
        avg_tracking_return = np.mean(active_return)

        information_ratio = 250 * avg_tracking_return / annual_te
        print('信息波动率：' + str(information_ratio))
        return information_ratio

    @staticmethod
    def downside_risk(account_daily_income_rate, standard_symbol_income_rate):
        if len(account_daily_income_rate) < 2:
            downside_risk = 0.
            annual_downside_risk = 0.
            return 0
        diff = np.array(account_daily_income_rate) - \
               np.array(standard_symbol_income_rate)
        diff[diff > 0] = 0.
        sum_mean_squares = np.sum(np.square(diff))
        # self._annual_downside_risk = (sum_mean_squares ** 0.5) * \
        #                              ((self._annual_factor / (len(self._portfolio) - 1)) ** 0.5)
        downside_risk = (sum_mean_squares / (len(diff) - 1)) ** 0.5
        annual_downside_risk = downside_risk * (250 ** 0.5)
        print('下行风险：' + str(downside_risk))
        print('年化下行风险：' + str(annual_downside_risk))
        return annual_downside_risk

    @staticmethod
    def tracking_error(account_daily_income_rate, standard_symbol_income_rate):
        if len(account_daily_income_rate) < 2:
            racking_error = 0.
            return 0

        active_return = np.array(account_daily_income_rate) - \
                        np.array(standard_symbol_income_rate)
        racking_error = active_return.std(ddof=1)
        print('跟踪误差：' + str(racking_error))
        return racking_error

    @staticmethod
    def annual_tracking_error(racking_error):
        annual_tracking_error = racking_error * (250 ** 0.5)
        print('年化跟踪误差：' + str(annual_tracking_error))
        return annual_tracking_error

    @staticmethod
    def sortino(ar,
                sr,
                downside_risk):

        if downside_risk == 0:
            sortino = None
            return sortino

        sortino = (ar - sr) / downside_risk
        print('索提诺比率:' + str(sortino))
        return sortino

    @staticmethod
    def avg_excess_return(account_daily_income_rate, daily_risk_free_rate):
        avg_excess_return = np.mean(
            np.array(account_daily_income_rate) -
            np.array(daily_risk_free_rate))
        return avg_excess_return

    @staticmethod
    def new_annual_return(end_account_rate, date_num):
        if date_num == 0:
            return 0
        if 1 + end_account_rate < 0:
            annual_return_value = -1 * math.pow(
                abs(end_account_rate) - 1, 1 / (date_num / 250)) - 1
        else:
            annual_return_value = math.pow(
                1 + end_account_rate, 1 / (date_num / 250)) - 1
        print('基准年化收益：%s' % str(annual_return_value))
        return annual_return_value

    @staticmethod
    def kama_ratio(strategy_profit_year, max_drawdown):
        if max_drawdown == 0:
            return None
        return strategy_profit_year / max_drawdown
