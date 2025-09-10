from panda_server.enums.feature_tag import FeatureTag
from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Header,
    status,
    BackgroundTasks,
)
import traceback
import logging
from typing import Optional, List
from panda_server.logic.workflow_save_logic import workflow_save_logic
from panda_server.logic.workflow_list_logic import workflow_list_logic
from panda_server.logic.workflow_get_logic import workflow_get_logic
from panda_server.logic.workflow_delete_logic import workflow_delete_logic
from panda_server.logic.workflow_run_logic import workflow_run_logic
from panda_server.logic.workflow_run_get_logic import workflow_run_get_logic
from panda_server.logic.workflow_logs_get_logic import workflow_logs_get_logic
from panda_server.logic.workflow_terminate_logic import workflow_terminate_logic
from panda_server.logic.workflow_output_logic import workflow_output_get_logic
from panda_server.logic.workflow_run_output_by_last_run_logic import (
    WorkflowNoLastRunException,
    WorkflowRunFailedException,
    workflow_run_output_by_last_run_logic,
    WorkflowRunningException,
)
from panda_server.models.base_api_response import BaseAPIResponse
from panda_server.models.run_workflow_response import RunWorkflowResponse
from panda_server.models.run_workflow_request import (
    RunWorkflowRequest,
    TerminateWorkflowRunRequest,
)
from panda_server.models.save_workflow_response import SaveWorkflowResponse
from panda_server.models.save_workflow_request import SaveWorkflowRequest
from panda_server.models.delete_workflow_request import DeleteWorkflowRequest
from panda_server.models.query_workflow_run_response import QueryWorkflowRunResponse
from panda_server.models.query_logs_response import QueryWorkflowLogsResponse
from panda_server.models.query_workflows_response import QueryWorkflowsResponse

# 获取 logger
logger = logging.getLogger(__name__)

# 创建路由实例，设置前缀和标签
router = APIRouter(prefix="/api/workflow", tags=["workflow"])

# 定义 collection 名称
COLLECTION_NAME = "workflow"


@router.post(
    "/save", response_model=SaveWorkflowResponse, status_code=status.HTTP_201_CREATED
)
async def save_workflow(
    request: SaveWorkflowRequest,
    user_id: str = Header(..., alias="uid", description="用户ID"),
) -> SaveWorkflowResponse:
    """
    创建/修改保存 workflow

    Args:
        request: 保存工作流的请求数据
        user_id: 从 headers 中获取的用户ID

    Returns:
        SaveWorkflowResponse: 包含工作流ID的响应
    """

    logger.info(f"/api/workflow/save, user_id: {user_id}")


    return await workflow_save_logic(request, user_id)


@router.get(
    "/all", response_model=QueryWorkflowsResponse, status_code=status.HTTP_200_OK
)
async def query_workflow(
    user_id: str = Header(..., alias="uid", description="用户ID"),
    limit: int = Query(10, description="每页返回的数量", ge=1, le=100),
    page: int = Query(1, description="第几页，从1开始", ge=1),
    filter: Optional[List[str]] = Query(
        default_factory=list,
        description="按 filter 筛选工作流, 可传入多个值，当前支持: backtest, signal, factor, trade",
    ),
) -> QueryWorkflowsResponse:
    """
    查询指定用户的所有 workflow (支持分页和feature_tag筛选)

    Args:
        user_id: 从 headers 中获取的用户ID
        limit: 每页返回的数量，默认10，最大100
        page: 第几页，从1开始，默认1
        filter: 按 filter 筛选工作流, 可传入多个值，当前支持: backtest, signal, factor, trade

    Returns:
        QueryWorkflowsResponse: 包含工作流列表和总数的响应
    """

    logger.info(f"/api/workflow/all, user_id: {user_id}")
    return await workflow_list_logic(user_id, limit, page, filter)


@router.get("/query", status_code=status.HTTP_200_OK)
async def get_workflow(
    workflow_id: str = Query(..., description="工作流ID"),
    user_id: str = Header(..., alias="uid", description="用户ID"),
):
    """
    获取指定 workflow

    Args:
        workflow_id: 工作流ID
        user_id: 从 headers 中获取的用户ID

    Returns:
        WorkflowModel: 工作流信息
    """
    try:
        return await workflow_get_logic(workflow_id, user_id)
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid workflow id format: {workflow_id}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"unexpected error in get_workflow: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the workflow",
        )


@router.delete("/delete", status_code=status.HTTP_200_OK)
async def delete_workflows(
    request: DeleteWorkflowRequest,
    user_id: str = Header(..., alias="uid", description="用户ID"),
):
    """
    批量删除工作流

    Args:
        request: 删除工作流请求，包含要删除的工作流ID列表
        user_id: 从 headers 中获取的用户ID

    Returns:
        BaseAPIResponse: 删除操作的结果
    """
    try:
        return await workflow_delete_logic(request, user_id)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求数据格式错误: {str(e)}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"批量删除工作流时发生意外错误: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除工作流时发生意外错误",
        )


@router.post(
    "/run", response_model=RunWorkflowResponse, status_code=status.HTTP_201_CREATED
)
async def run_workflow_api(
    background_tasks: BackgroundTasks,
    request: RunWorkflowRequest,
    user_id: str = Header(..., alias="uid", description="用户ID"),
    quantflow_auth: str = Header(..., alias="quantflow-auth", description="认证凭据"),
) -> RunWorkflowResponse:
    """
    运行指定 workflow

    Args:
        background_tasks: FastAPI 后台任务
        request: 运行工作流的请求数据
        user_id: 从 headers 中获取的用户ID
        quantflow_auth: 认证凭据

    Returns:
        RunWorkflowResponse: 运行工作流的响应

    TODO 稍后处理: 因为用户在运行过程中如果再次保存 workflow 会造成运行的代码版本不一致. 所以还要保存一个工作流的备份版本.
    """
    logger.info(f"/api/workflow/run, user_id: {user_id}")


    return await workflow_run_logic(background_tasks, request, user_id, quantflow_auth)


@router.get(
    "/run", response_model=QueryWorkflowRunResponse, status_code=status.HTTP_200_OK
)
async def query_workflow_run(
    user_id: str = Header(..., alias="uid", description="用户ID"),
    workflow_run_id: str = Query(..., description="工作流运行ID"),
) -> QueryWorkflowRunResponse:
    """
    查询指定 workflow run 运行状态

    Args:
        workflow_run_id: 工作流运行ID
        user_id: 从 headers 中获取的用户ID

    Returns:
        QueryWorkflowRunResponse: 查询工作流运行状态的响应
    """
    try:
        return await workflow_run_get_logic(workflow_run_id, user_id)
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid workflow run id format: {workflow_run_id}",
        )
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"unexpected error in query_workflow_run: {e}\n{stack_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while querying workflow run",
        )


@router.post(
    "/run/terminate",
    status_code=status.HTTP_200_OK,
)
async def terminate_workflow_run(
    request: TerminateWorkflowRunRequest,
    user_id: str = Header(..., alias="uid", description="用户ID"),
) -> BaseAPIResponse:
    """
    终止指定 workflow 运行

    Args:
        request: 终止工作流运行请求
        user_id: 从 headers 中获取的用户ID

    Returns:
        BaseAPIResponse: 终止操作的结果
    """
    return await workflow_terminate_logic(request, user_id)


@router.get("/run/output", status_code=status.HTTP_200_OK)
async def get_workflow_run_output(
    user_id: str = Header(..., alias="uid", description="用户ID"),
    output_obj_id: str = Query(..., description="输出对象ID"),
    locator: Optional[str] = Query(
        default="", description="字段检索条件, 如:'a.b.0'表示检索 a.b[0] 字段"
    ),
    page: int | None = Query(
        default=None, description="第几页，从1开始, 可选; 必须和 limit 一起使用", ge=1
    ),
    limit: int | None = Query(
        default=None,
        description="每页返回的数量, 可选, 取值范围1-100; 必须和 page 一起使用",
        ge=1,
        le=100,
    ),
) -> BaseAPIResponse:
    """
    根据 output_obj_id 获取输出对象
    Returns:
        BaseAPIResponse: 包含输出对象数据的响应
    """
    try:
        return await workflow_output_get_logic(
            output_obj_id, locator, user_id=user_id, page=page, limit=limit
        )
    except HTTPException:
        raise
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(
            f"Error getting workflow node output, id:{output_obj_id}, error:{e}, \nstack_trace:{stack_trace}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while getting the workflow node output",
        )


@router.get("/run/output/by-last-run", status_code=status.HTTP_200_OK)
async def get_workflow_run_output_by_last_run(
    user_id: str = Header(..., alias="uid", description="用户ID"),
    workflow_id: str = Query(
        description="workflow id, 将查询这个 workflow 最后一次运行的结果"
    ),
    feature_tag: FeatureTag = Query(
        description="feature tag, 将查询运行结果中特定的 feature tag 的输出"
    ),
    locator: Optional[str] = Query(
        default="", description="字段检索条件, 如:'a.b.0'表示检索 a.b[0] 字段"
    ),
    page: int | None = Query(
        default=None, description="第几页，从1开始, 可选; 必须和 limit 一起使用", ge=1
    ),
    limit: int | None = Query(
        default=None,
        description="每页返回的数量, 可选, 取值范围1-100; 必须和 page 一起使用",
        ge=1,
        le=100,
    ),
) -> BaseAPIResponse:
    """
    根据 workflow_id 获取最后一次运行的结果
    支持 feature tag 筛选
    """
    try:
        return await workflow_run_output_by_last_run_logic(
            workflow_id, feature_tag, locator, user_id=user_id, page=page, limit=limit
        )
    except (WorkflowRunningException, WorkflowRunFailedException, WorkflowNoLastRunException) as e:
        return BaseAPIResponse(code=e.code, message=e.message, data=None)
    except HTTPException:
        raise
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(
            f"Error getting workflow node output, id:{workflow_id}, error:{e}, \nstack_trace:{stack_trace}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while getting the workflow node output by last run",
        )


@router.get(
    "/run/log", response_model=QueryWorkflowLogsResponse, status_code=status.HTTP_200_OK
)
async def query_workflow_logs(
    workflow_run_id: str = Query(..., description="工作流运行ID"),
    work_node_id: Optional[str] = Query(None, description="工作节点ID"),
    log_level: Optional[str] = Query(None, description="日志等级过滤"),
    req_id: Optional[str] = Query(None, description="请求ID过滤 (x-request-id)"),
    backtest_id: Optional[str] = Query(None, description="回测ID过滤"),
    last_sequence: Optional[int] = Query(
        None, description="起始序列号，从该序列号开始获取日志（包含该序列号）"
    ),
    limit: int = Query(5, description="返回数量限制，默认5条", ge=1, le=1000),
    user_id: str = Header(..., alias="uid", description="用户ID"),
) -> QueryWorkflowLogsResponse:
    """
    查询工作流日志

    Args:
        workflow_run_id: 工作流运行ID，必填
        work_node_id: 工作节点ID，可选
        log_level: 日志等级过滤，可选
        last_sequence: 起始序列号，从该序列号开始获取日志（包含该序列号），可选
        limit: 返回数量限制，默认5条
        user_id: 从 headers 中获取的用户ID

    Returns:
        QueryWorkflowLogsResponse: 查询工作流日志的响应

    支持特性：
    - 支持以 last_sequence 向后分页（获取更新的日志）
    - 支持 limit 分页查询，默认5条
    - 支持以 workflow run id 为条件检索
    - 支持以 work node id 为条件检索
    - 支持以日志等级为条件检索

    排序说明：
    - 指定 workflow_run_id 时：严格按 sequence 升序排序，确保同一workflow内日志顺序正确
    - 跨 workflow 查询时：按 timestamp + sequence 排序，保证时间顺序的合理性

    分页说明：
    - 首次查询：不传 last_sequence，获取从序列号1开始的日志
    - 后续查询：传入上次结果的 next_sequence 作为 last_sequence，从该序列号开始获取
    - last_sequence=5 表示从序列号5开始获取（包含序列号5）
    - 返回的 next_sequence 是下次查询的起始序列号
    """
    return await workflow_logs_get_logic(
        user_id=user_id,
        workflow_run_id=workflow_run_id,
        work_node_id=work_node_id,
        log_level=log_level,
        req_id=req_id,
        backtest_id=backtest_id,
        last_sequence=last_sequence,
        limit=limit,
    )
