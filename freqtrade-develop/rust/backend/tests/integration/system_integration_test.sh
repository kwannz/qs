#!/bin/bash

# 系统集成测试脚本
# 完整的端到端测试

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 测试配置
API_URL=${API_URL:-"http://localhost:8080"}
WS_URL=${WS_URL:-"ws://localhost:8080"}
TEST_USER="test@trading.com"
TEST_PASS="Test123456!"

# 测试计数器
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 日志函数
log_info() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_TESTS++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 测试函数
run_test() {
    local test_name=$1
    local test_func=$2
    
    ((TOTAL_TESTS++))
    log_info "运行测试: $test_name"
    
    if $test_func; then
        log_success "$test_name"
    else
        log_error "$test_name"
    fi
}

# ==================== 测试用例 ====================

# 1. 健康检查测试
test_health_check() {
    response=$(curl -s -o /dev/null -w "%{http_code}" ${API_URL}/api/v1/health)
    [ "$response" = "200" ]
}

# 2. 认证测试
test_authentication() {
    # 登录
    response=$(curl -s -X POST ${API_URL}/api/v1/auth/login \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"${TEST_USER}\",\"password\":\"${TEST_PASS}\"}")
    
    # 提取token
    ACCESS_TOKEN=$(echo $response | jq -r '.accessToken')
    REFRESH_TOKEN=$(echo $response | jq -r '.refreshToken')
    
    [ ! -z "$ACCESS_TOKEN" ] && [ "$ACCESS_TOKEN" != "null" ]
}

# 3. 获取用户信息测试
test_get_profile() {
    response=$(curl -s -X GET ${API_URL}/api/v1/auth/profile \
        -H "Authorization: Bearer ${ACCESS_TOKEN}")
    
    user_id=$(echo $response | jq -r '.id')
    [ ! -z "$user_id" ] && [ "$user_id" != "null" ]
}

# 4. 市场数据测试
test_market_data() {
    # 获取行情
    response=$(curl -s -X GET ${API_URL}/api/v1/market/ticker/BTCUSDT \
        -H "Authorization: Bearer ${ACCESS_TOKEN}")
    
    last_price=$(echo $response | jq -r '.lastPrice')
    [ ! -z "$last_price" ] && [ "$last_price" != "null" ]
}

# 5. 获取K线数据测试
test_kline_data() {
    response=$(curl -s -X GET "${API_URL}/api/v1/market/klines?symbol=BTCUSDT&interval=1h&limit=100" \
        -H "Authorization: Bearer ${ACCESS_TOKEN}")
    
    kline_count=$(echo $response | jq '. | length')
    [ "$kline_count" -gt 0 ]
}

# 6. 订单测试
test_order_management() {
    # 创建订单
    response=$(curl -s -X POST ${API_URL}/api/v1/trading/orders \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{
            "symbol": "BTCUSDT",
            "side": "BUY",
            "orderType": "LIMIT",
            "quantity": 0.001,
            "price": 40000
        }')
    
    ORDER_ID=$(echo $response | jq -r '.id')
    
    if [ -z "$ORDER_ID" ] || [ "$ORDER_ID" = "null" ]; then
        return 1
    fi
    
    # 查询订单
    response=$(curl -s -X GET ${API_URL}/api/v1/trading/orders/${ORDER_ID} \
        -H "Authorization: Bearer ${ACCESS_TOKEN}")
    
    order_status=$(echo $response | jq -r '.status')
    
    # 取消订单
    curl -s -X DELETE ${API_URL}/api/v1/trading/orders/${ORDER_ID} \
        -H "Authorization: Bearer ${ACCESS_TOKEN}"
    
    [ ! -z "$order_status" ] && [ "$order_status" != "null" ]
}

# 7. 账户信息测试
test_account_info() {
    response=$(curl -s -X GET ${API_URL}/api/v1/trading/accounts \
        -H "Authorization: Bearer ${ACCESS_TOKEN}")
    
    balance=$(echo $response | jq -r '.balance')
    [ ! -z "$balance" ] && [ "$balance" != "null" ]
}

# 8. 策略测试
test_strategy_management() {
    # 创建策略
    response=$(curl -s -X POST ${API_URL}/api/v1/strategy/create \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Test Strategy",
            "type": "GRID",
            "symbols": ["BTCUSDT"],
            "parameters": {
                "gridLevels": 10,
                "upperPrice": 50000,
                "lowerPrice": 40000
            }
        }')
    
    STRATEGY_ID=$(echo $response | jq -r '.id')
    
    if [ -z "$STRATEGY_ID" ] || [ "$STRATEGY_ID" = "null" ]; then
        return 1
    fi
    
    # 删除策略
    curl -s -X DELETE ${API_URL}/api/v1/strategy/delete/${STRATEGY_ID} \
        -H "Authorization: Bearer ${ACCESS_TOKEN}"
    
    true
}

# 9. AI预测测试
test_ai_prediction() {
    response=$(curl -s -X POST ${API_URL}/api/v1/ai/predict \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{
            "symbol": "BTCUSDT",
            "timeframes": ["1h", "4h", "1d"]
        }')
    
    predictions=$(echo $response | jq -r '.predictions')
    [ ! -z "$predictions" ] && [ "$predictions" != "null" ]
}

# 10. 风险指标测试
test_risk_metrics() {
    response=$(curl -s -X GET ${API_URL}/api/v1/risk/metrics \
        -H "Authorization: Bearer ${ACCESS_TOKEN}")
    
    risk_score=$(echo $response | jq -r '.riskScore')
    [ ! -z "$risk_score" ] && [ "$risk_score" != "null" ]
}

# 11. WebSocket连接测试
test_websocket_connection() {
    # 使用wscat测试WebSocket连接
    if command -v wscat &> /dev/null; then
        timeout 5 wscat -c "${WS_URL}/ws?token=${ACCESS_TOKEN}" &
        ws_pid=$!
        sleep 2
        
        if ps -p $ws_pid > /dev/null; then
            kill $ws_pid 2>/dev/null
            return 0
        else
            return 1
        fi
    else
        log_warning "wscat未安装，跳过WebSocket测试"
        return 0
    fi
}

# 12. 并发测试
test_concurrent_requests() {
    local concurrent_count=10
    local success_count=0
    
    for i in $(seq 1 $concurrent_count); do
        (
            response=$(curl -s -o /dev/null -w "%{http_code}" \
                ${API_URL}/api/v1/market/ticker/BTCUSDT \
                -H "Authorization: Bearer ${ACCESS_TOKEN}")
            [ "$response" = "200" ] && echo "success"
        ) &
    done
    
    wait
    
    success_count=$(jobs -p | wc -l)
    [ $success_count -eq $concurrent_count ]
}

# 13. 错误处理测试
test_error_handling() {
    # 测试404
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        ${API_URL}/api/v1/nonexistent \
        -H "Authorization: Bearer ${ACCESS_TOKEN}")
    
    [ "$response" = "404" ]
}

# 14. Token刷新测试
test_token_refresh() {
    response=$(curl -s -X POST ${API_URL}/api/v1/auth/refresh \
        -H "Content-Type: application/json" \
        -d "{\"refreshToken\":\"${REFRESH_TOKEN}\"}")
    
    new_token=$(echo $response | jq -r '.accessToken')
    [ ! -z "$new_token" ] && [ "$new_token" != "null" ]
}

# 15. 性能测试
test_api_performance() {
    local total_time=0
    local test_count=10
    
    for i in $(seq 1 $test_count); do
        start_time=$(date +%s%N)
        curl -s ${API_URL}/api/v1/health > /dev/null
        end_time=$(date +%s%N)
        
        elapsed=$((($end_time - $start_time) / 1000000))
        total_time=$(($total_time + $elapsed))
    done
    
    avg_time=$(($total_time / $test_count))
    log_info "平均响应时间: ${avg_time}ms"
    
    # 检查是否小于200ms
    [ $avg_time -lt 200 ]
}

# ==================== 主流程 ====================

main() {
    log_info "开始系统集成测试"
    log_info "API URL: ${API_URL}"
    echo ""
    
    # 检查服务是否运行
    if ! curl -s ${API_URL}/api/v1/health > /dev/null; then
        log_error "服务未运行或无法访问"
        exit 1
    fi
    
    # 运行测试
    run_test "健康检查" test_health_check
    run_test "用户认证" test_authentication
    run_test "获取用户信息" test_get_profile
    run_test "市场数据" test_market_data
    run_test "K线数据" test_kline_data
    run_test "订单管理" test_order_management
    run_test "账户信息" test_account_info
    run_test "策略管理" test_strategy_management
    run_test "AI预测" test_ai_prediction
    run_test "风险指标" test_risk_metrics
    run_test "WebSocket连接" test_websocket_connection
    run_test "并发请求" test_concurrent_requests
    run_test "错误处理" test_error_handling
    run_test "Token刷新" test_token_refresh
    run_test "API性能" test_api_performance
    
    echo ""
    echo "======================================"
    echo "测试结果汇总"
    echo "======================================"
    echo -e "总测试数: ${TOTAL_TESTS}"
    echo -e "${GREEN}通过: ${PASSED_TESTS}${NC}"
    echo -e "${RED}失败: ${FAILED_TESTS}${NC}"
    echo ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "所有测试通过！"
        exit 0
    else
        log_error "有 ${FAILED_TESTS} 个测试失败"
        exit 1
    fi
}

# 运行主流程
main