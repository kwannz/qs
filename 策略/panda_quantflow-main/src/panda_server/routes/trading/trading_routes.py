import json
import traceback

from fastapi import APIRouter, HTTPException, status, Path, Query, Body
from typing import Optional, Dict, Any, List
from bson import ObjectId
from datetime import datetime, time, timezone

from fastapi.encoders import jsonable_encoder

from panda_server.logic.trading.real_trade_binding_logic import find_strategy_account
from panda_server.logic.trading.trad_constant import TradingConstant
from panda_server.config.database import mongodb
from panda_server.models.base_api_response import BaseAPIResponse
import logging
from panda_trading.models.TradeCollections import TradeCollections
from panda_trading.models.trading import FutureAccountModel, FutureAccountCreateModel, FutureAccountUpdateModel
from panda_trading.models.trading.trading_real_server import RealTradStrategyServerModel, PageResponse, \
    RegisterServerRequest, ServerStatus
from panda_trading.models.trading.trading_real_binding import RealTradeBindingCreate, RealTradeBindingModel
from common.connector.redis_client import RedisClient

# 获取 logger
logger = logging.getLogger(__name__)

# 创建路由实例，设置前缀和标签
router = APIRouter(prefix="/api/trading", tags=["trading"])
redis_client = RedisClient()

# 实盘交易表 collection 名称
FUTURE_ACCOUNT_COLLECTION = TradeCollections.FUTURE_ACCOUNT
REAL_TRAD_STRATEGY_COLLECTION = TradeCollections.REAL_TRADE_STRATEGY
REAL_TRADE_BINDING_COLLECTION = TradeCollections.REAL_TRADE_BINDING

@router.get("/startAccountMonitor")
async def start_account_monitor(
    strategy_id: int,
    account: str,
    account_type: int
):
    # Step 1: 查询 MongoDB 数据
    result = await find_strategy_account(strategy_id, account, account_type)
    if not result:
        raise HTTPException(status_code=400, detail="Account not found in MongoDB")

    monitor_server = result.get("monitorServer")
    if not monitor_server:
        raise HTTPException(status_code=500, detail="Monitor server not configured")

    # Step 2: 更新 Redis Hash 状态
    redis_key = TradingConstant.ACCOUNT_MONITOR_PROGRESS + account
    redis_client.setHashRedis(redis_key, {
        "status": "0",
        "update_time": str(int(time.time() * 1000)),
        "err_mes": ""
    })

    # Step 3: 发送 Redis 消息
    route_key = TradingConstant.ACCOUNT_MONITOR_SERVER + monitor_server
    message = {
        "type": "start_trade",
        "account": account,
        "account_type": str(account_type)
    }
    redis_client.public(route_key, json.dumps(message))

    return {"message": "Account monitor started successfully", "account": account}
@router.post("/restartMyStrategy/{strategy_id}", response_model=BaseAPIResponse)
async def restart_product_strategy(
    strategy_id: str = Path(..., description="产品策略ID")
):
    """
    重启实盘策略接口
    """
    try:
        # 获取集合
        collection = mongodb.get_collection(REAL_TRADE_BINDING_COLLECTION)

        # 1. 检查策略是否存在
        if not ObjectId.is_valid(strategy_id):
            raise HTTPException(status_code=400, detail="无效的策略ID格式")

        strategy_obj_id = ObjectId(strategy_id)
        strategy_record = await collection.find_one({
            "_id": strategy_obj_id,
            "is_deleted": {"$ne": 1}
        })

        if not strategy_record:
            raise HTTPException(status_code=404, detail="未找到对应策略记录")

        strategy_data = RealTradeBindingModel(**strategy_record)

        # 2. 更新 Redis 中策略状态为 "0"（准备启动）
        redis_key = f"{TradingConstant.REAL_TRADE_PROGRESS}{strategy_id}"
        redis_client.setHashRedis(redis_key, "status", "0")
        redis_client.setHashRedis(redis_key, "update_time", str(int(datetime.now().timestamp() * 1000)))
        redis_client.setHashRedis(redis_key, "err_mes", "")

        # 3. 获取策略服务器IP
        server_id = strategy_data.strategy_server
        if not ObjectId.is_valid(server_id):
            raise HTTPException(status_code=400, detail="策略服务器ID无效")

        server_collection = mongodb.get_collection("real_trad_strategy_server")
        server_record = await server_collection.find_one({"_id": ObjectId(server_id)})
        if not server_record:
            raise HTTPException(status_code=404, detail="策略服务器不存在")

        server_ip = server_record.get("server_ip")
        if not server_ip:
            raise HTTPException(status_code=400, detail="策略服务器无可用IP地址")

        # 4. 构建消息内容并发布到 Redis 主题
        content = {
            "type": "start_trade",
            "run_id": strategy_id
        }

        route_key = f"{TradingConstant.REAL_TRADE_SERVER}{server_ip}"
        redis_client.public(route_key, json.dumps(content))

        return BaseAPIResponse(
            code=200,
            message= "策略已发送启动指令",
            data= {
                "product_strategy_id": strategy_id,
                "server_ip": server_ip
            }
        )

    except Exception as e:
        logger.error(f"重启策略失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"重启策略失败: {str(e)}")

# === 添加策略实盘绑定 ===
@router.post("/bind/addMyStrategy2RealTrade", response_model=BaseAPIResponse)
async def add_strategy_to_real_trade(
        binding_data: RealTradeBindingCreate = Body(..., description="绑定信息")
):
    """
    添加策略到实盘交易绑定
    """
    try:
        collection = mongodb.get_collection(REAL_TRADE_BINDING_COLLECTION)

        # 校验1: 检查服务器是否存在
        server_collection = mongodb.get_collection(REAL_TRAD_STRATEGY_COLLECTION)

        server_exists = await server_collection.find_one({
            "server_ip": binding_data.strategy_server,
            "is_deleted": {"$ne": 1}
        })
        if not server_exists:
            raise HTTPException(
                status_code=404,
                detail="策略服务器不存在或已下线"
            )

        # 校验2: 检查是否已存在相同绑定
        existing_binding = await collection.find_one({
            "user_id": binding_data.user_id,
            "strategy_id": binding_data.strategy_id,
            "future_account": binding_data.future_account,
            "is_deleted": 0
        })
        if existing_binding:
            raise HTTPException(
                status_code=400,
                detail="该策略已绑定到此实盘账户"
            )

        # 创建完整文档
        now = datetime.now(tz=timezone.utc)
        binding_doc = RealTradeBindingModel(
            **binding_data.model_dump(),
            create_time=now,
            update_time=now
        ).model_dump(exclude={"id"})

        # 插入数据库
        result = await collection.insert_one(binding_doc)

        return BaseAPIResponse(
            code=200,
            message="策略实盘绑定成功",
            data={"binding_id": str(result.inserted_id)}
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"添加策略实盘绑定失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"添加策略实盘绑定失败: {str(e)}"
        )


# === 获取策略实盘绑定分页列表 ===
@router.get("/bind/getRealTradeBindings", response_model=BaseAPIResponse)
async def get_real_trade_bindings(
        page: int = Query(1, ge=1, description="页码"),
        size: int = Query(10, ge=1, le=100, description="每页数量"),
        user_id: Optional[str] = Query(None, description="用户ID过滤"),
        strategy_id: Optional[str] = Query(None, description="策略ID过滤"),
        future_account: Optional[str] = Query(None, description="实盘账号模糊查询")
):
    """
    获取策略实盘绑定分页列表 实盘服务器状态:-2异常、-1待运行、0停止、1运行中
    """
    try:
        collection = mongodb.get_collection(REAL_TRADE_BINDING_COLLECTION)
        tradeServerCollection = mongodb.get_collection(REAL_TRAD_STRATEGY_COLLECTION)
        # 构建查询条件
        query = {"is_deleted": 0}  # 只查询未删除的

        if user_id:
            query["user_id"] = user_id
        if strategy_id:
            query["strategy_id"] = strategy_id
        if future_account:
            query["future_account"] = {"$regex": future_account, "$options": "i"}

        # 分页计算
        skip = (page - 1) * size
        total = await collection.count_documents(query)

        # 查询并转换结果
        cursor = collection.find(query).skip(skip).limit(size)
        items = [jsonable_encoder(RealTradeBindingModel(**doc)) async for doc in cursor]
        #
        # 提取所有 strategy_server IP 地址
        server_ips = list({item["strategy_server"] for item in items if "strategy_server" in item})

        # 批量查询服务器状态
        server_query = {"server_ip": {"$in": server_ips}}
        servers_cursor = tradeServerCollection.find(server_query, {"server_ip": 1, "status": 1})
        servers_data = {s["server_ip"]: s["status"] async for s in servers_cursor}

        # 添加 status 字段到每个绑定记录
        result_items = []
        for item in items:
            binding_model = RealTradeBindingModel(**item)
            item_dict = jsonable_encoder(binding_model)
            item_dict = binding_model.model_dump(by_alias=True)
            item_dict["strategy_server_status"] = servers_data.get(item.get("strategy_server"), -1)  # 默认异常状态
            result_items.append(item_dict)

        return BaseAPIResponse(
            code=200,
            message="查询成功",
            data={
                "total": total,
                "page": page,
                "size": size,
                "items": result_items,
                "description": f"实盘服务器状态:-2异常、-1待运行、0停止、1运行中"
            }
        )

    except Exception as e:
        logger.error(f"获取绑定列表失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"获取绑定列表失败: {str(e)}"
        )


# === 更新策略实盘绑定 ===
@router.put("/bind/updateRealTradeBinding/{binding_id}", response_model=BaseAPIResponse)
async def update_real_trade_binding(
        binding_id: str = Path(..., description="绑定记录ID"),
        update_data: RealTradeBindingCreate = Body(..., description="更新数据")
):
    """
    更新策略实盘绑定信息
    """
    try:
        # 验证ID格式
        if not ObjectId.is_valid(binding_id):
            raise HTTPException(
                status_code=400,
                detail="无效的绑定ID格式"
            )

        collection = mongodb.get_collection(REAL_TRADE_BINDING_COLLECTION)
        obj_id = ObjectId(binding_id)

        # 检查绑定是否存在
        existing_binding = await collection.find_one({"_id": obj_id, "is_deleted": 0})
        if not existing_binding:
            raise HTTPException(
                status_code=404,
                detail="绑定记录不存在或已删除"
            )

        # 校验服务器是否存在
        server_collection = mongodb.get_collection(REAL_TRAD_STRATEGY_COLLECTION)
        if not ObjectId.is_valid(update_data.strategy_server):
            raise HTTPException(
                status_code=400,
                detail="无效的服务器ID格式"
            )

        server_exists = await server_collection.find_one({
            "_id": ObjectId(update_data.strategy_server),
            "is_deleted": {"$ne": 1}
        })
        if not server_exists:
            raise HTTPException(
                status_code=404,
                detail="策略服务器不存在或已下线"
            )

        # 更新数据
        update_doc = update_data.dict()
        update_doc["update_time"] = datetime.utcnow()

        result = await collection.update_one(
            {"_id": obj_id},
            {"$set": update_doc}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=500,
                detail="绑定更新失败"
            )

        return BaseAPIResponse(
            code=200,
            message="绑定更新成功",
            data={"binding_id": binding_id}
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"更新绑定失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"更新绑定失败: {str(e)}"
        )


# === 删除策略实盘绑定 (软删除) ===
@router.delete("/bind/deleteRealTradeBinding/{binding_id}", response_model=BaseAPIResponse)
async def delete_real_trade_binding(
        binding_id: str = Path(..., description="绑定记录ID")
):
    """
    删除策略实盘绑定 (软删除)
    """
    try:
        # 验证ID格式
        if not ObjectId.is_valid(binding_id):
            raise HTTPException(
                status_code=400,
                detail="无效的绑定ID格式"
            )

        collection = mongodb.get_collection(REAL_TRADE_BINDING_COLLECTION)
        obj_id = ObjectId(binding_id)

        # 检查绑定是否存在
        existing_binding = await collection.find_one({"_id": obj_id, "is_deleted": 0})
        if not existing_binding:
            raise HTTPException(
                status_code=404,
                detail="绑定记录不存在或已删除"
            )

        # 执行软删除
        result = await collection.update_one(
            {"_id": obj_id},
            {"$set": {"is_deleted": 1, "update_time": datetime.utcnow()}}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=500,
                detail="删除绑定失败"
            )

        return BaseAPIResponse(
            code=200,
            message="绑定删除成功",
            data={"binding_id": binding_id}
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"删除绑定失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"删除绑定失败: {str(e)}"
        )


# === 服务器状态变更接口 ===
@router.post("/setServerStatus", response_model=BaseAPIResponse)
async def set_server_status(
        data: ServerStatus
):
    """
    设置服务器状态 实盘服务器状态:-2异常、-1待运行、0停止、1运行中
    """
    try:
        # 验证ID格式
        if not ObjectId.is_valid(data.id):
            raise HTTPException(
                status_code=400,
                detail="无效的服务器ID格式"
            )

        # 验证状态值
        if data.status not in (0, 1):
            raise HTTPException(
                status_code=400,
                detail="状态值必须是0(下线)或1(上线)"
            )

        collection = mongodb.get_collection(REAL_TRAD_STRATEGY_COLLECTION)
        obj_id = ObjectId(data.id)

        # 检查服务器是否存在
        server = await collection.find_one({"_id": obj_id})
        if not server:
            raise HTTPException(
                status_code=404,
                detail="服务器不存在"
            )

        # 更新状态
        result = await collection.update_one(
            {"_id": obj_id},
            {"$set": {"status": data.status}}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=304,
                detail=f"状态无更新"
            )

        return BaseAPIResponse(
            code=200,
            message=f"服务器已{'上线' if data.status == 1 else '下线'}",
            data={"id": data.id}
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"状态更新失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"状态更新失败: {str(e)}"
        )


# === 服务器上线接口 (快捷方式) ===
@router.post("/activateServer/{server_id}", response_model=BaseAPIResponse)
async def activate_server(
        server_id: str = Path(..., description="服务器ID")
):
    """
    激活服务器 (设置为上线状态)
    """
    return await set_server_status(ServerStatus(id=server_id, status=1))


# === 服务器下线接口 (快捷方式) ===
@router.post("/deactivateServer/{server_id}", response_model=BaseAPIResponse)
async def deactivate_server(
        server_id: str = Path(..., description="服务器ID")
):
    """
    停用服务器 (设置为下线状态)
    """
    return await set_server_status(ServerStatus(id=server_id, status=0))


# === 服务器存在性检查接口 ===
@router.get("/checkServerExist/{server_id}", response_model=BaseAPIResponse)
async def check_server_exist(
        server_id: str = Path(..., description="服务器ID")
):
    """
    检查服务器是否存在并返回当前状态
    """
    try:
        # 验证ID格式
        if not ObjectId.is_valid(server_id):
            return BaseAPIResponse(
                code=200,
                message="ID格式无效",
                data={"exists": False, "status": None}
            )

        collection = mongodb.get_collection(REAL_TRAD_STRATEGY_COLLECTION)
        obj_id = ObjectId(server_id)

        # 查询服务器
        server = await collection.find_one({"_id": obj_id})

        if server:
            # 获取当前状态 (默认为1-上线)
            status = server.get("status", 1)
            return BaseAPIResponse(
                code=200,
                message="服务器存在",
                data={"exists": True, "status": status}
            )
        else:
            return BaseAPIResponse(
                code=200,
                message="服务器不存在",
                data={"exists": False, "status": None}
            )

    except Exception as e:
        logger.error(f"存在性检查失败: {str(e)}\n{traceback.format_exc()}")
        return BaseAPIResponse(
            code=500,
            message="服务器检查失败",
            data={"exists": False, "status": None}
        )


@router.post("/registerRealTradStrategyServer", response_model=BaseAPIResponse)
async def register_real_trad_strategy_server(
        request: RegisterServerRequest
):
    """
    注册新的策略服务器
    """
    try:
        collection = mongodb.get_collection(REAL_TRAD_STRATEGY_COLLECTION)

        # 校验1: 检查IP是否已存在
        existing_ip = await collection.find_one(
            {"server_ip": request.server_ip}
        )
        if existing_ip:
            raise HTTPException(
                status_code=400,
                detail=f"IP地址 {request.server_ip} 已被注册"
            )
        # 校验2: 检查名称是否已存在
        existing_name = await collection.find_one(
            {"name": request.name}
        )
        if existing_name:
            raise HTTPException(
                status_code=400,
                detail=f"服务器名称 {request.name} 已被使用"
            )
        # 创建新文档
        server_doc = {
            "server_ip": request.server_ip,
            "name": request.name,
            "remark": request.remark or ""
        }
        # 插入数据库
        result = await collection.insert_one(server_doc)
        return BaseAPIResponse(
            code=200,
            message="服务器注册成功",
            data={
                "server_id": str(result.inserted_id)
            }
        )
    except Exception as e:
        logger.error(f"服务器注册失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"服务器注册失败: {str(e)}"
        )


@router.get("/getRealTradStrategyServerPageList", response_model=BaseAPIResponse)
async def get_real_trad_strategy_page_list(
        page: int = Query(1, ge=1, description="当前页码"),
        size: int = Query(10, ge=1, le=100, description="每页数量"),
        server_ip: Optional[str] = Query(None, description="服务器IP模糊查询"),
        name: Optional[str] = Query(None, description="服务器名称模糊查询"),
        remark: Optional[str] = Query(None, description="备注信息模糊查询")
):
    """
    获取实盘策略服务器分页列表
    支持按server_ip/name/remark进行模糊查询
    """
    try:
        # 获取MongoDB集合 (根据实际项目替换获取方式)
        collection = mongodb.get_collection(REAL_TRAD_STRATEGY_COLLECTION)  # type: AsyncIOMotorCollection

        # 构建查询条件
        query = {}
        if server_ip:
            query["server_ip"] = {"$regex": server_ip, "$options": "i"}  # 不区分大小写的模糊匹配
        if name:
            query["name"] = {"$regex": name, "$options": "i"}
        if remark:
            query["remark"] = {"$regex": remark, "$options": "i"}

        # 计算分页参数
        skip = (page - 1) * size

        # 执行分页查询
        cursor = collection.find(query).skip(skip).limit(size)
        items = [RealTradStrategyServerModel(**doc) async for doc in cursor]

        # 获取总数
        total = await collection.count_documents(query)

        return BaseAPIResponse(
            code=200,
            message="查询成功",
            data=PageResponse(
                total=total,
                page=page,
                size=size,
                items=items
            )
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"服务器内部错误: {str(e)}"
        )


@router.post("/createFutureAccount", response_model=BaseAPIResponse)
async def create_future_account(account: FutureAccountCreateModel):
    """
    创建期货账户接口
    """
    try:
        collection = mongodb.get_collection(FUTURE_ACCOUNT_COLLECTION)

        # 将 Pydantic 模型转为字典，并添加创建时间
        account_dict = account.model_dump(by_alias=True, exclude_unset=True)
        account_dict["gmt_create"] = datetime.now()
        account_dict["gmt_modified"] = datetime.now()

        # 插入数据库
        result = await collection.insert_one(account_dict)

        # 返回成功响应
        return BaseAPIResponse(
            code=200,
            message="期货账户创建成功",
            data={"id": str(result.inserted_id)}
        )

    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in create_future_account: {e}\n{stack_trace}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/getFutureAccountPageList", response_model=BaseAPIResponse)
async def get_future_account_page_list(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    account_id: Optional[str] = None,
    name: Optional[str] = None
):
    """
    获取期货账户分页列表
    支持按 account_id 和 name 进行模糊查询
    """
    try:
        collection = mongodb.get_collection(FUTURE_ACCOUNT_COLLECTION)

        # 构建查询条件
        query = {"is_deleted": {"$ne": 1}}  # 默认过滤未删除的数据
        if account_id:
            query["account_id"] = {"$regex": account_id, "$options": "i"}
        if name:
            query["name"] = {"$regex": name, "$options": "i"}

        # 分页计算
        skip = (page - 1) * size
        cursor = collection.find(query).skip(skip).limit(size)

        # 转换结果为模型对象
        total = await collection.count_documents(query)
        items = [FutureAccountModel(**item) for item in await cursor.to_list(length=size)]

        # 返回分页结果
        return BaseAPIResponse(
            code=200,
            message="获取期货账户列表成功",
            data={
                "total": total,
                "page": page,
                "size": size,
                "items": items
            }
        )

    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in get_future_account_page_list: {e}\n{stack_trace}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/getFutureAccountById/{account_id}", response_model=BaseAPIResponse)
async def get_future_account_by_id(account_id: str):
    """
    根据 MongoDB ObjectId 获取单个账户信息
    """
    try:
        collection = mongodb.get_collection(FUTURE_ACCOUNT_COLLECTION)
        query = {"_id": ObjectId(account_id), "is_deleted": {"$ne": 1}}

        account = await collection.find_one(query)
        if not account:
            raise HTTPException(status_code=404, detail="账户不存在或已删除")

        return BaseAPIResponse(
            code=200,
            message="获取账户信息成功",
            data=FutureAccountModel(**account)
        )

    except HTTPException:
        raise
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in get_future_account_by_id: {e}\n{stack_trace}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.put("/updateFutureAccount/{account_id}", response_model=BaseAPIResponse)
async def update_future_account(account_id: str, update_data: FutureAccountUpdateModel):
    """
    更新期货账户信息
    """
    try:
        collection = mongodb.get_collection(FUTURE_ACCOUNT_COLLECTION)

        # 查询是否存在该账户
        existing = await collection.find_one({"_id": ObjectId(account_id)})
        if not existing or existing.get("is_deleted") == 1:
            raise HTTPException(status_code=404, detail="账户不存在或已删除")

        # 构建更新内容
        update_dict = update_data.model_dump(by_alias=True, exclude_unset=True)
        update_dict["gmt_modified"] = datetime.now()

        # 执行更新
        await collection.update_one(
            {"_id": ObjectId(account_id)},
            {"$set": update_dict}
        )

        return BaseAPIResponse(
            code=200,
            message="账户信息更新成功"
        )

    except HTTPException:
        raise
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in update_future_account: {e}\n{stack_trace}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.delete("/deleteFutureAccount/{account_id}", response_model=BaseAPIResponse)
async def delete_future_account(account_id: str):
    """
    删除账户（软删除）
    """
    try:
        collection = mongodb.get_collection(FUTURE_ACCOUNT_COLLECTION)

        existing = await collection.find_one({"_id": ObjectId(account_id)})
        if not existing or existing.get("is_deleted") == 1:
            raise HTTPException(status_code=404, detail="账户不存在或已删除")

        # 执行软删除
        await collection.update_one(
            {"_id": ObjectId(account_id)},
            {"$set": {"is_deleted": 1, "gmt_modified": datetime.now()}}
        )

        return BaseAPIResponse(
            code=200,
            message="账户删除成功（软删除）"
        )

    except HTTPException:
        raise
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Unexpected error in delete_future_account: {e}\n{stack_trace}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

