# ============================================================
# MODEL PARAMETERS — The control knobs
# ============================================================

from langchain_ollama import ChatOllama
import asyncio


PromptFromUser = input("Enter your Input : ")



# ------------------------------------ temperature: Creativity vs Consistency ----------------------------------------
# 0 = always same answer (deterministic)
# 0.7 = balanced (default)
# 1.0 = creative, varied
# 2.0 = wild, often nonsensical

factualllm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0
)
creativellm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=2
)
print(f"Factual LLM : {factualllm.invoke(PromptFromUser).content}")
print("__________________________________________________________")
print(f"Creative LLM : {creativellm.invoke(PromptFromUser).content}")







# --------------------------------------- max_tokens: Maximum response length ----------------------------------------
# 1 token ≈ 3/4 of a word
# This is a MAX, not a target — model stops early if it finishes its thought

short_llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0,
    max_token=50
)     # Brief answers

long_llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=2,
    max_token=300

)     # Detailed answers

print("Short:", short_llm.invoke(PromptFromUser).content)
print("Long:", long_llm.invoke(PromptFromUser).content)





# ============================================================
# PARAMETER CHEAT SHEET
# ============================================================
#
# | Task                  | temperature | max_tokens |
# |-----------------------|-------------|------------|
# | Code generation       | 0           | 2000       |
# | Factual Q&A           | 0           | 500        |
# | Chatbot               | 0.7         | 1000       |
# | Creative writing      | 1.0         | 2000       |
# | Data extraction/JSON  | 0           | 500        |
# | Brainstorming         | 1.0         | 1000       |
#
# ============================================================



# ------------------------------------- top_p: Another way to control randomness -------------------------------------
# Controls how many word options to consider
# 0.1 = only top candidates, 1.0 = all candidates
# Rule: Adjust temperature OR top_p, not both

# High temperature but narrow selection = controlled creativity
llm_focused = ChatOllama(model="gemma3:latest", base_url="http://localhost:11434", temperature=1.0, top_p=0.1)

# --- timeout & max_retries: Reliability ---
reliable_llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    timeout=30,       # Give up after 30 seconds
    max_retries=3,    # Retry up to 3 times on failure
)

# --- The typical setup for most use cases ---
llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.7,
    max_tokens=1000,
    timeout=30,
    max_retries=2,
)


