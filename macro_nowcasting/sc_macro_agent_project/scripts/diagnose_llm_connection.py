"""LLM 连接测试。若无 API Key，验证 mock 降级路径。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sc_macro_agent.llm import LLMClient

client = LLMClient()
print(f"Mock mode: {client.is_mock}")

response = client.chat(
    system="你是一个经济分析助手。回答简洁准确。",
    user="用一句话概括四川省2024年GDP累计同比增速的趋势。",
)
print(f"Response: {response}")
print(f"Usage: {client.get_usage_stats()}")
