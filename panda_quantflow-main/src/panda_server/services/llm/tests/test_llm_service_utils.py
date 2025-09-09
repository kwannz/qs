import uuid
from panda_server.services.llm.utils import LLMServiceUtils
import pytest
import dotenv

pytestmark = pytest.mark.asyncio

class TestLLMServiceUtils:
    """测试 TestLLMServiceUtils 类的功能"""

    @classmethod
    def setup_class(cls):
        """在整个测试类开始前执行一次"""
        dotenv.load_dotenv()
        # 为这一组测试生成唯一的UUID
        cls.test_run_id = str(uuid.uuid4())[:8]
        print(f"测试组ID: {cls.test_run_id}")

    async def _setup_test(self, test_suffix=""):
        """异步测试初始化"""
        # 使用类级别的UUID，同一组测试共享
        self.test_user_id = f"test_user_{self.test_run_id}{test_suffix}"
        self.service = LLMServiceUtils()  
        self.test_user_message = """Tell me a joke, and return the joke in json format"""
  
        # 连接数据库
        try:
            from panda_server.config.database import mongodb
            await mongodb.connect_db()
            print("数据库连接成功")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            pytest.skip("数据库连接失败")

    @pytest.mark.asyncio
    async def test_process_message_new_session_non_stream(self):
        """测试创建新会话并处理非流式消息"""
        await self._setup_test("_non_stream")

        session_id, result = await self.service.process_message(
            user_id=self.test_user_id,
            user_message=self.test_user_message,
            stream=False,
            json_mode=True
        )
        print(f"Session ID: {session_id}")
        assert len(session_id) > 0
        print(f"Chat completion result: {result}")
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_process_message_stream_mode(self):
        """测试流式消息处理"""
        await self._setup_test("_stream")

        session_id, stream_response = await self.service.process_message(
            user_id=self.test_user_id,
            user_message=self.test_user_message,
            stream=True,
            json_mode=True
        )
        print(f"Session ID: {session_id}")
        assert len(session_id) > 0
        result = []
        async for chunk in stream_response:
            result.append(chunk)

        print(f"Chat completion stream result: {''.join(result)}")
        assert len(result) > 0


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
