# ============================================================
# THE PROBLEM — LLMs always return raw text
# ============================================================

# What the LLM gives you:
# "The movie Inception has a rating of 9.2 out of 10. It's a mind-bending thriller."

# What your app actually needs:
# {"title": "Inception", "rating": 9.2, "summary": "A mind-bending thriller"}

# You can't put a paragraph into a database.
# You can't use a sentence as an API response.
# Output Parsers convert raw text → structured Python data.


from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.5
)


# ============================================================
# 1. StrOutputParser — Just Give Me the Text
#    Strips AIMessage wrapper, returns plain string
# ============================================================

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
response = llm.invoke("What is Python")

