import logging
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
project_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 logger
logger = logging.getLogger(__name__)
from common.config import ProjectConfig

from panda_trading.trading_route.manager.real_trade_manager import RealTradeManager
app = FastAPI()
# 定义请求模型
class TradeRequest(BaseModel):
    run_id: str
    type: str  # "start_trade" 或 "kill_run_trade"


# 初始化交易管理器
trade_manager = RealTradeManager()

@app.post("/start-trade")
async def start_trade(request: TradeRequest):
    if request.type != 'start_trade':
        raise HTTPException(status_code=400, detail="Invalid request type")

    try:
        trade_manager.start_trade(request.run_id)
        return {"status": "success", "message": f"Started trade for run_id: {request.run_id}"}
    except Exception as e:
        traceback_str = traceback.format_exc()
        return {"status": "error", "message": str(e), "traceback": traceback_str}

@app.post("/stop-trade")
async def stop_trade(request: TradeRequest):
    if request.type != 'kill_run_trade':
        raise HTTPException(status_code=400, detail="Invalid request type")

    try:
        trade_manager.kill_run_trade(request.run_id)
        return {"status": "success", "message": f"Stopped trade for run_id: {request.run_id}"}
    except Exception as e:
        traceback_str = traceback.format_exc()
        return {"status": "error", "message": str(e), "traceback": traceback_str}

@app.get("/")
async def get():
    return HTMLResponse("panda trading started")

if __name__ == '__main__':
    host_ip = ProjectConfig.get_config_parser(project_dir).get('server_ip', 'ip')
    uvicorn.run(app, host=host_ip, port=8300)





