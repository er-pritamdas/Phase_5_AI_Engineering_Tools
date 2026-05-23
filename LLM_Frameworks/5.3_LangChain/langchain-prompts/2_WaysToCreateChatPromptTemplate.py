from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage



# ============================================================
#  FOUR WAYS to Create ChatPromptTemplate
# ============================================================


# --- Way 1: Tuple shorthand (most common, cleanest) ---
prompt_v1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role}."),
    ("user", "{question}")
])




# --- Way 2: Using Message objects directly ---
# Use this when some messages are fixed (no variables)
prompt_v2 = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),   # Fixed — no variables
    ("user", "{question}")                                   # Dynamic — has variable
])





# --- Way 3: Simple single-message template ---
# When you only need a user message, no system prompt
prompt_v3 = ChatPromptTemplate.from_template("Explain {topic} simply.")
# Creates just one HumanMessage






# --- Way 4: Multi-turn conversation template ---
# Fake a conversation history in the template itself
prompt_v4 = ChatPromptTemplate.from_messages([
    ("system", "You are a tutor."),
    ("user", "My name is {name}"),
    ("ai", "Hello {name}! How can I help?"),    # Fake AI history
    ("user", "{question}")                       # The actual question
])




# ============================================================
# FIXED vs DYNAMIC Messages
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Python expert."),       # FIXED — same every time
    ("user", "Explain {topic} with examples.")    # DYNAMIC — changes each call
])

# Only provide the dynamic variable
result = prompt.invoke({"topic": "decorators"})
print(result)
# SystemMessage stays fixed, HumanMessage changes
