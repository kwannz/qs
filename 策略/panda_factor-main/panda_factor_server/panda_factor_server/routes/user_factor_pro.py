import os
from typing import Optional
from fastapi import APIRouter, Query, Depends, Header, HTTPException
from panda_factor_server.services.user_factor_service import *

_db_handler = DatabaseHandler(config)

router = APIRouter()


def api_key_auth(x_api_key: Optional[str] = Header(default=None)):
    """简单 API Key 认证。若未配置 API_KEY，则不启用校验。"""
    expected = os.getenv("API_KEY") or config.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


@router.get("/hello")
async def hello_route(_: bool = Depends(api_key_auth)):
    return hello()

@router.get("/user_factor_list")
async def user_factor_list_route(
    user_id: str,
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=10, ge=1, le=100, description="每页数量"),
    sort_field: str = Query(default="created_at", description="排序字段，支持updated_at、created_at、return_ratio、sharpe_ratio、maximum_drawdown、IC、IR"),
    sort_order: str = Query(default="desc", description="排序方式，asc升序，desc降序"),
    _: bool = Depends(api_key_auth)
):
    """
    获取用户因子列表
    :param user_id: 用户ID
    :param page: 页码，从1开始
    :param page_size: 每页数量，默认10，最大100
    :param sort_field: 排序字段
    :param sort_order: 排序方式，asc或desc
    :return: 因子列表，包含基本信息和性能指标
    """
    return get_user_factor_list(user_id, page, page_size, sort_field, sort_order)

@router.post("/create_factor")
async def create_factor_route(factor: CreateFactorRequest, _: bool = Depends(api_key_auth)):
    return create_factor(factor)

@router.get("/delete_factor")
async def delete_user_factor_route(factor_id: str, _: bool = Depends(api_key_auth)):
    return delete_factor(factor_id)

@router.post("/update_factor")
async def update_factor_route(factor: CreateFactorRequest, factor_id: str, _: bool = Depends(api_key_auth)):
    return update_factor(factor, factor_id)

@router.get("/query_factor")
async def query_factor_route(factor_id: str, _: bool = Depends(api_key_auth)):
    return query_factor(factor_id)
@router.get("/query_factor_status")
async def query_factor_status_route(factor_id: str, _: bool = Depends(api_key_auth)):
    return query_factor_status(factor_id)

@router.get("/run_factor")
async def run_factor_route(factor_id: str, _: bool = Depends(api_key_auth)):
    return run_factor(factor_id, is_thread=True)

@router.get("/query_task_status")
async def query_task_status_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_task_status(task_id)

@router.get("/query_factor_excess_chart")
async def query_factor_excess_chart_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_factor_excess_chart(task_id)

@router.get("/query_factor_analysis_data")
async def query_factor_analysis_data_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_factor_analysis_data(task_id)

@router.get("/query_group_return_analysis")
async def query_group_return_analysis_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_group_return_analysis(task_id)

@router.get("/query_ic_decay_chart")
async def query_ic_decay_chart_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_ic_decay_chart(task_id)

@router.get("/query_ic_density_chart")
async def query_ic_density_chart_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_ic_density_chart(task_id)

@router.get("/query_ic_self_correlation_chart")
async def query_ic_self_correlation_chart_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_ic_self_correlation_chart(task_id)

@router.get("/query_ic_sequence_chart")
async def query_ic_sequence_chart_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_ic_sequence_chart(task_id)

@router.get("/query_last_date_top_factor")
async def query_last_date_top_factor_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_last_date_top_factor(task_id)

@router.get("/query_one_group_data")
async def query_one_group_data_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_one_group_data(task_id)

@router.get("/query_rank_ic_decay_chart")
async def query_rank_ic_decay_chart_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_rank_ic_decay_chart(task_id)

@router.get("/query_rank_ic_density_chart")
async def query_rank_ic_density_chart_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_rank_ic_density_chart(task_id)

@router.get("/query_rank_ic_self_correlation_chart")
async def query_rank_ic_self_correlation_chart_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_rank_ic_self_correlation_chart(task_id)

@router.get("/query_rank_ic_sequence_chart")
async def query_rank_ic_sequence_chart_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_rank_ic_sequence_chart(task_id)

@router.get("/query_return_chart")
async def query_return_chart_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_return_chart(task_id)

@router.get("/query_simple_return_chart")
async def query_simple_return_chart_route(task_id: str, _: bool = Depends(api_key_auth)):
    return query_simple_return_chart(task_id)

# --- 新增 RESTful 风格路由（兼容保留旧路由） ---
@router.post("/factors", dependencies=[Depends(api_key_auth)])
async def factors_create(factor: CreateFactorRequest):
    return create_factor(factor)

@router.get("/factors/{factor_id}", dependencies=[Depends(api_key_auth)])
async def factors_get(factor_id: str):
    return query_factor(factor_id)

@router.put("/factors/{factor_id}", dependencies=[Depends(api_key_auth)])
async def factors_update(factor_id: str, factor: CreateFactorRequest):
    return update_factor(factor, factor_id)

@router.delete("/factors/{factor_id}", dependencies=[Depends(api_key_auth)])
async def factors_delete(factor_id: str):
    return delete_factor(factor_id)

@router.post("/factors/{factor_id}/run", dependencies=[Depends(api_key_auth)])
async def factors_run(factor_id: str, is_thread: bool = True):
    return run_factor(factor_id, is_thread=is_thread)

@router.get("/factors/{factor_id}/status", dependencies=[Depends(api_key_auth)])
async def factors_status(factor_id: str):
    return query_factor_status(factor_id)

@router.get("/tasks/{task_id}", dependencies=[Depends(api_key_auth)])
async def tasks_get(task_id: str):
    return query_task_status(task_id)

@router.get("/tasks/{task_id}/logs", dependencies=[Depends(api_key_auth)])
async def tasks_logs(task_id: str, last_log_id: str = None):
    return get_task_logs(task_id, last_log_id=last_log_id)

@router.get("/tasks/{task_id}/charts/excess", dependencies=[Depends(api_key_auth)])
async def charts_excess(task_id: str):
    return query_factor_excess_chart(task_id)

@router.get("/tasks/{task_id}/charts/analysis", dependencies=[Depends(api_key_auth)])
async def charts_analysis(task_id: str):
    return query_factor_analysis_data(task_id)

@router.get("/tasks/{task_id}/charts/group-return", dependencies=[Depends(api_key_auth)])
async def charts_group_return(task_id: str):
    return query_group_return_analysis(task_id)

@router.get("/tasks/{task_id}/charts/ic/decay", dependencies=[Depends(api_key_auth)])
async def charts_ic_decay(task_id: str):
    return query_ic_decay_chart(task_id)

@router.get("/tasks/{task_id}/charts/ic/density", dependencies=[Depends(api_key_auth)])
async def charts_ic_density(task_id: str):
    return query_ic_density_chart(task_id)

@router.get("/tasks/{task_id}/charts/ic/self-correlation", dependencies=[Depends(api_key_auth)])
async def charts_ic_selfcorr(task_id: str):
    return query_ic_self_correlation_chart(task_id)

@router.get("/tasks/{task_id}/charts/ic/sequence", dependencies=[Depends(api_key_auth)])
async def charts_ic_sequence(task_id: str):
    return query_ic_sequence_chart(task_id)

@router.get("/tasks/{task_id}/charts/rank-ic/decay", dependencies=[Depends(api_key_auth)])
async def charts_rank_ic_decay(task_id: str):
    return query_rank_ic_decay_chart(task_id)

@router.get("/tasks/{task_id}/charts/rank-ic/density", dependencies=[Depends(api_key_auth)])
async def charts_rank_ic_density(task_id: str):
    return query_rank_ic_density_chart(task_id)

@router.get("/tasks/{task_id}/charts/rank-ic/self-correlation", dependencies=[Depends(api_key_auth)])
async def charts_rank_ic_selfcorr(task_id: str):
    return query_rank_ic_self_correlation_chart(task_id)

@router.get("/tasks/{task_id}/charts/rank-ic/sequence", dependencies=[Depends(api_key_auth)])
async def charts_rank_ic_sequence(task_id: str):
    return query_rank_ic_sequence_chart(task_id)

@router.get("/tasks/{task_id}/charts/return", dependencies=[Depends(api_key_auth)])
async def charts_return(task_id: str):
    return query_return_chart(task_id)

@router.get("/tasks/{task_id}/charts/return-simple", dependencies=[Depends(api_key_auth)])
async def charts_return_simple(task_id: str):
    return query_simple_return_chart(task_id)

@router.get("/task_logs")
async def get_task_logs_route(task_id: str, last_log_id: str = None):
    return get_task_logs(task_id, last_log_id=last_log_id)
