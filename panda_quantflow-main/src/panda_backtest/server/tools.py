import asyncio
import logging

import inspect
import linecache
import os
import re

import aiohttp
import async_timeout
from json2html import json2html
from pandas import DataFrame

from common.connector.mongodb_handler import DatabaseHandler
from common.config.config import config

def python2html(data_convert):
    html_result = None
    if data_convert is None:
        return html_result
    if isinstance(data_convert, dict) or isinstance(data_convert, list) or isinstance(data_convert, tuple):
        html_result = json2html.convert(json=data_convert, table_attributes='border="0" cellspacing="3" cellpadding="3"')
    elif isinstance(data_convert, DataFrame):
        html_result = data_convert.to_html(border=0)
    else:
        if not isinstance(data_convert, str):
            data_convert = str(data_convert)
        html_result = data_convert.replace('\n', '<br/>')
        html_result = html_result.replace(' ', '&nbsp;')
        if html_result.startswith('<class'):
            html_result = re.sub(r'^<class(.*?)>', '', html_result, 1)
        if html_result.startswith('<br/>'):
            html_result = html_result.replace('<br/>', '', 1)
    return html_result

def export_dataframe_to_file(data_frame, strategy_id, date, time, run_type):
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    asyncio.get_event_loop().run_until_complete(save_dataframe_to_mongo(data_frame, strategy_id, date, time, run_type))

async def save_dataframe_to_mongo(data_frame, strategy_id, date, time, run_type):
    db = DatabaseHandler(config)
    collection_name = "data_frame_record"
    collection = db[collection_name]
    mongo_data = dict()
    mongo_data['back_id'] = str(strategy_id)
    mongo_data['date'] = str(date)
    mongo_data['time'] = str(time)
    mongo_data['run_type'] = str(run_type)
    # mongo_data['data'] = str(data_frame)
    mongo_data['data'] = python2html(data_frame)
    collection.insert(mongo_data)

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(test())
