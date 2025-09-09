from enum import Enum
from panda_server.config.env import DEEPSEEK_API_KEY

class LLMModelType(str, Enum):
    """
    LLM model type enumeration
    """
    DeepSeek_V3 = "deepseek-chat"
    DeepSeek_R1 = "deepseek-reasoner"


# Model configuration mapping display names to model details
MODEL_CONFIG = {
    "DeepSeek-V3": {
        "model": LLMModelType.DeepSeek_V3,
        "base_url": "https://api.deepseek.com/v1",
        "api_key_name": DEEPSEEK_API_KEY,
        "version": "DeepSeek-V3-0324"
    },
    "DeepSeek-R1": {
        "model": LLMModelType.DeepSeek_R1,
        "base_url": "https://api.deepseek.com/v1",
        "api_key_name": DEEPSEEK_API_KEY,
        "version": "DeepSeek-R1-0528"
    }
} 