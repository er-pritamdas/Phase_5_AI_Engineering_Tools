# ============================================================
# MESSAGE TYPES — The 3 roles in a conversation
# ============================================================

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.5
)


# ------ System Message ------------
system = SystemMessage(content="You are a Professional Gym Coach")

# ------ Human Message --------------
human = HumanMessage(content="My Name is Pritam")

# ---------AI Message -----------------
AI = AIMessage(content="Nice to meet you pritam!!")


messages = [system, human, AI, HumanMessage(content="Do you know my name ?")]
response = llm.invoke(messages)
print(response.content)




# ============================================================
#    ChatPromptTemplate — Messages with Roles (USE THIS!)
#    Creates a list of messages: System, Human, AI
# ============================================================
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are an expert in {domain}. Be concise"),
        ("user", "{question}")
    ]
)
# The tuple shorthand automatically converts:
# ("system", "...")    → SystemMessage
# ("user", "...")      → HumanMessage
# ("human", "...")     → HumanMessage  (same as "user")
# ("assistant", "...") → AIMessage
# ("ai", "...")        → AIMessage     (same as "assistant")

message = prompt.invoke(
    {
        "domain": "Python",
        "question": "What is Decorator ?"
    }
)
print(message)