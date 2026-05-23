# ============================================================
# THE PROBLEM — Why not just use f-strings?
# ============================================================

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.5
)

topic = input("Enter the topic you want to learn : ")
response = llm.invoke(f"Explain {topic} in simple terms")
print(response.content)

# But f-strings CAN'T:
# - Plug into LCEL chains with | operator
# - Validate that all variables are provided
# - Handle message roles (system/user/ai)
# - Include few-shot examples
# - Be saved/loaded from files






# LangChain PromptTemplates solve all of this.
# ============================================================
#    PromptTemplate — Plain Text (Old Style)
#    Creates a single text string with {placeholders}
# ============================================================

from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Explain {topic} in {style} language"
)
prompt = template.invoke({"topic": "Python", "style": "simple"})
print(prompt) # Explain Python in simple language
print(llm.invoke(prompt).content)


# What if you forget a variable?
# result = template.invoke({"topic": "Python"})
# ❌ ERROR! Missing variable 'style'
# PromptTemplate validates for you — f-strings wouldn't catch this


