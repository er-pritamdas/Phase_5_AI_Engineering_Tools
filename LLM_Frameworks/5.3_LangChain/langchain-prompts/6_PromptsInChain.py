from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.5
)


# --------------------------------------------
# Without Chains
# --------------------------------------------

# template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(content="You are a expert Fitness Coach"),
#         ("human", "{question}")
#     ]
# )

# prompt = template.invoke(
#     {
#         "question" : "Suggest me five sources of food rich in Protien"
#     }
# )

# response = llm.invoke(prompt)
# print(response.content)



# --------------------------------------------
# With Chains
# --------------------------------------------

template = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are a expert Fitness Coach"),
        ("human", "{question}")
    ]
)

chain = template  | llm

response = chain.invoke(
    {
        "question" : "Suggest me five sources of food rich in Protien"
    }
)

print(response.content)
