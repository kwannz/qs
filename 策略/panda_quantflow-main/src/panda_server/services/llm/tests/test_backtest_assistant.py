import json
from panda_server.services.llm.agents.backtest_assistant import BacktestAssistant
import pytest
import pytest_asyncio
import dotenv

pytestmark = pytest.mark.asyncio


class TestBacktestAssistant:
    """测试 BacktestAssistant 类的功能"""

    @classmethod
    def setup_class(cls):
        """在整个测试类开始前执行一次"""
        dotenv.load_dotenv()
        cls.test_user_id = "test_user_id"

    @pytest_asyncio.fixture(autouse=True)
    async def setup_db(self):
        """每个测试方法执行前连接数据库"""
        from panda_server.config.database import mongodb
        try:
            print("开始连接数据库...")
            await mongodb.connect_db()
            print(f"数据库连接成功")
            yield
        except Exception as e:
            print(f"数据库连接失败: {e}")
            raise
        finally:
            print("开始关闭数据库连接...")
            await mongodb.close_db()
            print("数据库连接已关闭")

    # @pytest.mark.skip()
    @pytest.mark.asyncio
    async def test_process_message_from_empty_code(self):
        old_code = ""
        user_message = "写一个股票('000001.SZ')的布林带策略"
        assistant = BacktestAssistant()

        session_id, result = await assistant.process_message(
            user_id=self.test_user_id,
            user_message=user_message,
            original_user_code=old_code,
        )

        print(f"Session ID: {session_id}")
        print(f"result:\n{result}")
        assert len(session_id) > 0
        assert len(result) > 0

        result_json = json.loads(result)
        assert "code" in result_json
        assert "explanation" in result_json
        
        print(f"explanation:\n{result_json['explanation']}")
        print(f"code:\n{result_json['code']}")

    @pytest.mark.skip()
    @pytest.mark.asyncio
    async def test_multiple_rounds_chat(self):
        old_code = ""
        user_message = "写一个股票('000001.SZ')的布林带策略"
        assistant = BacktestAssistant()

        session_id, result = await assistant.process_message(
            user_id=self.test_user_id,
            user_message=user_message,
            original_user_code=old_code,
        )

        print(f"[Round 1] Session ID: {session_id}")
        assert len(session_id) > 0
        print(f"[Round 1] result: {result}")
        assert len(result) > 0
        result_json = json.loads(result)
        assert "code" in result_json
        assert "explanation" in result_json

        user_message = "代码逻辑不变,改为英文注释"
        session_id2, result = await assistant.process_message(
            session_id=session_id,
            user_id=self.test_user_id,
            user_message=user_message,
            original_user_code=result_json["code"],
        )

        assert session_id == session_id2
        print(f"[Round 2] result: {result}")
        assert len(result) > 0
        result_json = json.loads(result)
        assert "code" in result_json
        assert "explanation" in result_json

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
