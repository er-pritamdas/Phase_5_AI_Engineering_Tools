from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama



llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.5
)

#############################################################################
# ------------------------------- Without Memory ---------------------------
#############################################################################

# while True:
#     print("\n------------------------------------------------\n")
#     prompt = input("YOU : ")

#     if prompt == "exit":
#         break

#     print("\n--------------------- AI ---------------------\n")
#     for chunk in llm.stream(prompt):
#         print(chunk.content, end="", flush=True)


#############################################################################
# ----------------- With Memory (Without Message Tagging)--------------------
#############################################################################

# ChaHistory = []

# while True:
#     print("\n------------------------------------------------\n")
#     prompt = input("YOU : ")
#     ChaHistory.append(prompt)

#     if prompt == "exit":
#         break

#     print("\n--------------------- AI ---------------------\n")
#     response = llm.invoke(ChaHistory)
#     ChaHistory.append(response.content)
#     print(response.content)

# print("----------------------------- Chat history ----------------------------")
# for i in ChaHistory:
#     print(f"# - {i}")



#############################################################################
# ----------------- With Memory (With Message Tagging)--------------------
#############################################################################

ChaHistory = [
    SystemMessage(content="You are a helpful AI Assistant")
]

while True:
    print("\n------------------------------------------------\n")
    prompt = input("YOU : ")
    ChaHistory.append(HumanMessage(content=prompt))

    if prompt == "exit":
        break

    print("\n--------------------- AI ---------------------\n")
    response = llm.invoke(ChaHistory)
    ChaHistory.append(AIMessage(content=response.content))
    print(response.content)

print("----------------------------- Chat history ----------------------------")
for i in ChaHistory:
    print(f"# {i.type}- {i.content}")
# print(ChaHistory)