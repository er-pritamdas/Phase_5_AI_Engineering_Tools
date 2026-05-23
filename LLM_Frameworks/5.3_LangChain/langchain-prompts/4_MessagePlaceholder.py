from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# 7. MessagesPlaceholder — Inject Conversation History
#    The key to building memory-aware prompts
# ============================================================


# Create a prompt with a slot for conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # Past messages go HERE
    ("user", "{question}")
])

# Pass conversation history as a variable
messages = prompt.invoke({
    "chat_history": [
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a programming language."),
        HumanMessage(content="Who created it?"),
        AIMessage(content="Guido van Rossum."),
    ],
    "question": "What year was it created?"
})

# What the model sees:
for msg in messages.messages:
    print(f"  {msg.type}: {msg.content}")
# system: You are a helpful assistant.
# human: What is Python?
# ai: Python is a programming language.
# human: Who created it?
# ai: Guido van Rossum.
# human: What year was it created?

# The model sees the FULL conversation and can answer based on context!
# This is how you'll implement memory later.