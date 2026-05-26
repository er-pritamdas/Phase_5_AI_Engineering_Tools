from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.5
)

# ============================================================
# REUSABLE PROMPTS — Define once, use everywhere
# ============================================================

# Define a reusable prompt template
code_explainer = ChatPromptTemplate.from_messages([
    ("system", "You are a senior {language} developer. Explain code clearly with comments."),
    ("user", "Explain this code:\n```\n{code}\n```")
])


# Python code
result = llm.invoke({
    "language": "Python",
    "code": "[x**2 for x in range(10)]"
})
print("Python explanation:", result)

# JavaScript code — same template!
result = llm.invoke({
    "language": "JavaScript",
    "code": "const sum = arr.reduce((a, b) => a + b, 0);"
})
print("JS explanation:", result)