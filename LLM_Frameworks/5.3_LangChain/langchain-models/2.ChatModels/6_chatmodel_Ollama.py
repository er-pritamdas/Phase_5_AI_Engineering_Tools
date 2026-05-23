from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.5
)

# PromptFromUser = input("Enter your prompt : ")


# ------------------------------------------- 5 Ways to call a Model ----------------------------------

# 1. Invoke : Standard Call, Send Input, Get Full Response
# --------- 
# response = llm.invoke(PromptFromUser)
# print(response.content)



# 2. Stream : Token by Token, Like Someone is Typing
#----------
# for chunk in llm.stream(PromptFromUser):
#     print(chunk.content, end="", flush=True)


# 3. Batch : Multiple Inputs at once (runs in Parallels)
#----------
responses = llm.batch([
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
])

for index, Response in enumerate(responses):
    print(f"Response for {index+1} : {Response.content}")
    print("##############################################")