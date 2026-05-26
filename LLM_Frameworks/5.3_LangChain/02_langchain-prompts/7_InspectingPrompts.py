from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# INSPECTING PROMPTS — See what variables are needed
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("user", "Tell me about {topic} in {style} style.")
])

# See what variables this prompt expects
print("Required variables:", prompt.input_variables)
# ['role', 'topic', 'style']

# See the message templates
print("Messages:", prompt.messages)


# ============================================================
# SUMMARY CHEAT SHEET
# ============================================================
#
# | What                              | When to Use                           |
# |-----------------------------------|---------------------------------------|
# | PromptTemplate                    | Plain text prompts (rare, old style)  |
# | ChatPromptTemplate.from_messages()| Multi-role prompts (most common)      |
# | ChatPromptTemplate.from_template()| Quick single-message prompt           |
# | MessagesPlaceholder               | Inject conversation history           |
# | ("system", "...")                  | Set AI personality/rules              |
# | ("user", "...") or ("human", ...) | User's input                          |
# | ("ai", "...") or ("assistant",...)| AI's previous response                |
#
# Key insight: PromptTemplates are RUNNABLES
# They plug into LCEL chains with | just like models and parsers
#
# prompt | llm | parser   ← This only works because prompt is a Runnable!
#
# ============================================================