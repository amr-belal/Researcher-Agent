from langchain_openai import ChatOpenAI
import os 
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )