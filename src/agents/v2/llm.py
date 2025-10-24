from langchain_ollama import ChatOllama

from config.values import GPT_OSS

model = ChatOllama(
    model=GPT_OSS,
    temperature=0.5,
    timeout=10,
    max_tokens=1000,
)
