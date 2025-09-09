#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/7/9 下午7:44
# @Author : wlb
# @File   : mysql_client.py
# @desc   :
import pymysql
import logging

from dbutils.pooled_db import PooledDB
from pymysql.cursors import DictCursor

from common.config.config import config

class MysqlClient(object):

    @classmethod
    def get_mysql_client(cls):
        """
        获取业务mongodb
        :return:
        """
        db_host = config['MYSQL_HOST']
        db_port = config['MYSQL_PORT']
        user = config['MYSQL_USER']
        password = config['MYSQL_PASSWORD']
        database = config['MYSQL_DATABASE']
        mysql_client = SunriseMySqlClient(db_host, db_port, user, password, database)
        return mysql_client

class BasePymysqlPool(object):
    def __init__(self, host, port, user, pwd, db):
        self.db_host = host
        self.db_port = port
        self.user = user
        self.password = pwd
        self.db = db
        self.conn = None
        self.cursor = None

class SunriseMySqlClient(BasePymysqlPool):
    """
    MYSQL数据库对象，负责产生数据库连接 , 此类中的连接采用连接池实现获取连接对象：conn = Mysql.getConn()
            释放连接对象;conn.close()或del conn
    """
    # 连接池对象
    __pool = None

    def __init__(self, host, port, user, pwd, db):
        super(SunriseMySqlClient, self).__init__(host, port, user, pwd, db)
        # 数据库构造函数，从连接池中取出连接，并生成操作游标
        self._conn = self.__getConn()
        self._cursor = self._conn.cursor()

    def __getConn(self):
        """
        @summary: 静态方法，从连接池中取出连接
        @return MySQLdb.connection
        """
        if SunriseMySqlClient.__pool is None:
            SunriseMySqlClient.__pool = PooledDB(creator=pymysql,  # 使用链接数据库的模块
                                                 # 链接池中最多共享的链接数量，0和None表示全部共享。PS: 无用，因为pymysql和MySQLdb等模块的 threadsafety都为1，所有值无论设置为多少，_maxcached永远为0，所以永远是所有链接都共享。
                                                 # ping MySQL服务端，检查是否服务可用。# 如：0 = None = never, 1 = default = whenever it is requested, 2 = when a cursor is created, 4 = when a query is executed, 7 = always
                                                 host=self.db_host,
                                                 port=self.db_port,
                                                 user=self.user,
                                                 passwd=self.password,
                                                 database=self.db,
                                                 charset='utf8',
                                                 # 返回字典类型,
                                                 cursorclass=DictCursor)
        return SunriseMySqlClient.__pool.connection()

    def getconn(self):
        self._conn = self.__getConn()
        self._cursor = self._conn.cursor()

    def getAll(self, sql, param=None):
        """
        @summary: 执行查询，并取出所有结果集
        @param sql:查询ＳＱＬ，如果有查询条件，请只指定条件列表，并将条件值使用参数[param]传递进来
        @param param: 可选参数，条件列表值（元组/列表）
        @return: result list(字典对象)/boolean 查询到的结果集
        """
        self.getconn()
        try:
            if param is None:
                count = self._cursor.execute(sql)

            else:
                count = self._cursor.execute(sql, param)
            if count > 0:
                result = self._cursor.fetchall()
            else:
                result = False
            self.dispose()
            return result
        except Exception as e:
            print("error_msg:", e.args)
            self.dispose()
            return None

    def getOne(self, sql, param=None):
        """
        @summary: 执行查询，并取出第一条
        @param sql:查询ＳＱＬ，如果有查询条件，请只指定条件列表，并将条件值使用参数[param]传递进来
        @param param: 可选参数，条件列表值（元组/列表）
        @return: result list/boolean 查询到的结果集
        """
        self.getconn()
        try:
            if param is None:
                count = self._cursor.execute(sql)
            else:
                count = self._cursor.execute(sql, param)
            if count > 0:
                result = self._cursor.fetchone()
            else:
                result = False
            self.dispose()
            return result
        except Exception as e:
            print("error_msg:", e.args)
            self.dispose()
            return None

    def getMany(self, sql, num, param=None):
        """
        @summary: 执行查询，并取出num条结果
        @param sql:查询ＳＱＬ，如果有查询条件，请只指定条件列表，并将条件值使用参数[param]传递进来
        @param num:取得的结果条数
        @param param: 可选参数，条件列表值（元组/列表）
        @return: result list/boolean 查询到的结果集
        """
        self.getconn()
        try:
            if param is None:
                count = self._cursor.execute(sql)
            else:
                count = self._cursor.execute(sql, param)
            if count > 0:
                result = self._cursor.fetchmany(num)
            else:
                result = False
            self.dispose()
            return result
        except Exception as e:
            print("error_msg:", e.args)
            self.dispose()
            return None

    def insertMany(self, sql, values):
        """
        @summary: 向数据表插入多条记录
        @param sql:要插入的ＳＱＬ格式
        @param values:要插入的记录数据tuple(tuple)/list[list]
        @return: count 受影响的行数
        """
        self.getconn()
        try:
            count = self._cursor.executemany(sql, values)
            self.dispose()
            return count
        except Exception as e:
            print("error_msg:", e.args)
            self.dispose()
            return None

    def __query(self, sql, param=None):
        try:
            self.getconn()
            if param is None:
                count = self._cursor.execute(sql)
            else:
                count = self._cursor.execute(sql, param)
            self.dispose()
            return count
        except Exception as e:
            print("error_msg:", e.args)
            self.dispose()
            return None

    def update(self, sql, param=None):
        """
        @summary: 更新数据表记录
        @param sql: ＳＱＬ格式及条件，使用(%s,%s)
        @param param: 要更新的  值 tuple/list
        @return: count 受影响的行数
        """
        return self.__query(sql, param)

    def insert(self, sql, param=None):
        """
        @summary: 更新数据表记录
        @param sql: ＳＱＬ格式及条件，使用(%s,%s)
        @param param: 要更新的  值 tuple/list
        @return: count 受影响的行数
        """
        return self.__query(sql, param)

    def delete(self, sql, param=None):
        """
        @summary: 删除数据表记录
        @param sql: ＳＱＬ格式及条件，使用(%s,%s)
        @param param: 要删除的条件 值 tuple/list
        @return: count 受影响的行数
        """
        return self.__query(sql, param)

    def begin(self):
        """
        @summary: 开启事务
        """
        self._conn.autocommit(0)

    def end(self, option='commit'):
        """
        @summary: 结束事务
        """
        if option == 'commit':
            self._conn.commit()
        else:
            self._conn.rollback()

    def dispose(self, isEnd=1):
        """
        @summary: 释放连接池资源
        """
        if isEnd == 1:
            self.end('commit')
        else:
            self.end('rollback')
        self._cursor.close()
        self._conn.close()
