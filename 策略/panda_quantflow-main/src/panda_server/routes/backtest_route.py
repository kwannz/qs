from fastapi import APIRouter, HTTPException, status, Query
from typing import Optional
import traceback
import logging

from panda_server.logic.backtest.backtest_backtest_get_logic import backtest_backtest_get_logic
from panda_server.logic.backtest.backtest_account_get_logic import backtest_account_get_logic
from panda_server.logic.backtest.backtest_position_get_logic import backtest_position_get_logic
from panda_server.logic.backtest.backtest_profit_get_logic import backtest_profit_get_logic
from panda_server.logic.backtest.backtest_trade_get_logic import backtest_trade_get_logic
from panda_server.logic.backtest.backtest_user_strategy_log_get_logic import backtest_user_strategy_log_get_logic
from panda_server.models.backtest.query_backtest_response import QueryBacktestBacktestResponse
from panda_server.models.backtest.query_account_response import QueryBacktestAccountListResponse
from panda_server.models.backtest.query_position_response import QueryBacktestPositionListResponse
from panda_server.models.backtest.query_profit_response import QueryBacktestProfitListResponse
from panda_server.models.backtest.query_trade_response import QueryBacktestTradeListResponse
from panda_server.models.backtest.query_user_strategy_log_response import QueryBacktestUserStrategyLogListResponse

# 获取 logger
logger = logging.getLogger(__name__)

# 创建路由实例，设置前缀和标签
router = APIRouter(
    prefix="/api/backtest",
    tags=["backtest"]
)

@router.get(
    "/backtest", response_model=QueryBacktestBacktestResponse,status_code=status.HTTP_200_OK)
async def get_backtest(
    back_id: str = Query(..., description="回测ID")
) -> QueryBacktestBacktestResponse:
    """
    根据回测ID获取回测结果
    Args:
        back_id: 回测ID (查询参数)
    Returns:
        回测结果详细信息，如果ID不存在则返回错误
    """
    try:
        return await backtest_backtest_get_logic(back_id)
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid backtest id format: {back_id}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in get_backtest_by_id: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the backtest result",
        )


@router.get("/account", response_model=QueryBacktestAccountListResponse,status_code=status.HTTP_200_OK)
async def get_backtest_account(
    back_id: str = Query(..., description="回测ID"),
    page: int = 1,
    page_size: int = 10
) -> QueryBacktestAccountListResponse:
    """
    根据回测ID获取回测账户信息，支持分页
    Args:
        back_id: 回测ID (查询参数)
        page: 当前页码，默认为1
        page_size: 每页数据条数，默认为10
    Returns:
        回测账户信息列表及分页信息
    """
    try:
        return await backtest_account_get_logic(back_id, page, page_size)
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid backtest id format: {back_id}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in get_backtest_account: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the backtest account result",
        )


@router.get("/position", response_model=QueryBacktestPositionListResponse, status_code=status.HTTP_200_OK)
async def get_backtest_positions(
    back_id: str = Query(..., description="回测ID"),
    page: int = 1,
    page_size: int = 10,
    date: Optional[str] = None
) -> QueryBacktestPositionListResponse:
    """
    根据回测ID获取回测持仓信息，支持分页
    Args:
        back_id: 回测ID (查询参数)
        page: 当前页码，默认为1
        page_size: 每页数据条数，默认为10
        date: 可选的日期参数，格式为 "YYYY-MM-DD"
    Returns:
        回测持仓信息列表及分页信息
    """
    try:
        return await backtest_position_get_logic(back_id, page, page_size, date)
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid backtest id format: {back_id}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in get_backtest_positions: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the backtest position result",
        )

@router.get("/profit", response_model=QueryBacktestProfitListResponse, status_code=status.HTTP_200_OK)
async def get_backtest_profits(
    back_id: str = Query(..., description="回测ID"),
    page: int = 1,
    page_size: int = 10
) -> QueryBacktestProfitListResponse:
    """
    根据回测ID获取回测收益信息，支持分页
    Args:
        back_id: 回测ID (查询参数)
        page: 当前页码，默认为1
        page_size: 每页数据条数，默认为10
    Returns:
        回测收益信息列表及分页信息
    """
    try:
        return await backtest_profit_get_logic(back_id, page, page_size)
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid backtest id format: {back_id}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in get_backtest_profits: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the backtest profit result",
        )

@router.get("/trade", response_model=QueryBacktestTradeListResponse, status_code=status.HTTP_200_OK)
async def get_backtest_trade(
    back_id: str = Query(..., description="回测ID"),
    page: int = 1,
    page_size: int = 10
) -> QueryBacktestTradeListResponse:
    """
    根据回测ID获取回测交易信息，支持分页
    Args:
        back_id: 回测ID (查询参数)
        page: 当前页码，默认为1
        page_size: 每页数据条数，默认为10
    Returns:
        回测交易信息列表及分页信息
    """
    try:
        return await backtest_trade_get_logic(back_id, page, page_size)
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid backtest id format: {back_id}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in get_backtest_trade: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the backtest trade result",
        )

@router.get("/userstrategylog", response_model=QueryBacktestUserStrategyLogListResponse)
async def get_user_strategy_logs(
    relation_id: str,
    last_sort: int = None,
    limit: int = 20
) -> QueryBacktestUserStrategyLogListResponse:
    """
    根据 relation_id 获取 panda_user_strategy_log 记录，支持基于 sort 字段的游标式分页，并返回游标信息
    """
    try:
        return await backtest_user_strategy_log_get_logic(relation_id, last_sort, limit)
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid relation_id format: {relation_id}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in get_user_strategy_logs: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the user strategy log result",
        ) 