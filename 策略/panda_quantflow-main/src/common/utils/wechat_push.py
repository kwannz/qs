# encoding:utf-8
import requests
import logging

import json

def wechat_push(title:str,content:str):
    token = '7ad1907b7aa042d181046391b03f7a22'  # 在pushpush网站中可以找到
    data = {
        "token": token,
        "title": title,
        "content": content
    }
    url = 'http://www.pushplus.plus/send'
    body = json.dumps(data).encode(encoding='utf-8')
    headers = {'Content-Type': 'application/json'}
    requests.post(url, data=body, headers=headers)

def send_wechat(title:str,content:str):
    import requests
    token = '7ad1907b7aa042d181046391b03f7a22'  # 在pushpush网站中可以找到
    template = 'markdown'
    topic = '18503883862'  # 群组编码
    url = f"http://www.pushplus.plus/send?token={token}&title={title}&content={content}&template={template}&topic={topic}"
    r = requests.get(url=url)
    print(r.text)
