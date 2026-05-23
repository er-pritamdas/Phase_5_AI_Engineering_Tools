# ============================================================
# UTILITY METHODS — Extra powers on any model
# ============================================================

from langchain_ollama import ChatOllama
import asyncio


PromptFromUser = input("Enter your Input : ")


# ----------------------------------- .with_fallbacks() — Backup model if primary fails ---------------------------------------

mainLLM = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
)

# backupLLM = ChatOllama(
#     model="llama3.1:8b",
#     base_url="http://localhost:11434",
# )
# SafeLLM = mainLLM.with_fallbacks([backupLLM]) # If gemma3 goes down, automatically tries llama3
# response = SafeLLM.invoke(PromptFromUser)
# print(response.content)



# ------------------------------------------ .with_retry() — Retry on failure -------------------------------------------------
# reliable_llm = SafeLLM.with_retry(stop_after_attempt=3) # If API Call Fails, retry up to 3 times automatically



# ----------------------------------- .with_structured_output() — Force structured response ------------------------------------
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="One sentence summary")

structured_llm = mainLLM.with_structured_output(MovieReview)
result = structured_llm.invoke(PromptFromUser)
print(f"Title: {result.title}")        # "Inception"
print(f"Rating: {result.rating}")      # 9.2
print(f"Summary: {result.summary}")    # "A mind-bending thriller about..."