import pytest
import dotenv
from panda_server.services.llm.base.llm_service import LLMService
from panda_server.services.llm.models.message_model import Message

pytestmark = pytest.mark.asyncio


class TestLLMService:
    """测试 TestLLMService 类的功能"""

    @classmethod
    def setup_class(cls):
        """在整个测试类开始前执行一次"""
        dotenv.load_dotenv()
        cls.system_prompt = "You are a helpful assistant."
        cls.llm = LLMService(cls.system_prompt)

    async def test_chat_completion_success(self):
        """测试 chat_completion 成功调用"""

        llm = LLMService(self.system_prompt)
        messages = [Message(role="user", content="Tell me a joke")]

        result = await llm.chat_completion(messages)

        print(f"Chat completion result: {result}")

        assert len(result) > 0

    async def test_chat_completion_stream(self):
        """测试 chat_completion_stream 流式响应"""
        llm = LLMService(self.system_prompt)
        messages = [Message(role="user", content="Tell me a joke")]

        chunks = []
        async for chunk in llm.chat_completion_stream(messages):
            chunks.append(chunk)

        final_result = "".join(chunks)
        print(f"Chat completion stream result: {final_result}")

        assert len(final_result) > 0

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
