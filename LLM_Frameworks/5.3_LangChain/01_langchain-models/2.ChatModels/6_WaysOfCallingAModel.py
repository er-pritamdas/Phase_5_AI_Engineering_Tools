from langchain_ollama import ChatOllama
import asyncio

llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.5
)

PromptFromUser = input("Enter your prompt : ")


# ------------------------------------------- 5 Ways to call a Model ----------------------------------

###########################################################
# 1. Invoke : Standard Call, Send Input, Get Full Response
###########################################################
response = llm.invoke(PromptFromUser)
print(response.content)
print(type(response))                # <class 'langchain_core.messages.AIMessage'>
print(response.response_metadata)    # {'token_usage': {...}, 'model_name': 'gpt-4o'}
print(response.id)                   # Unique ID for this response



###########################################################
# 2. Stream : Token by Token, Like Someone is Typing
###########################################################
for chunk in llm.stream(PromptFromUser):
    print(chunk.content, end="", flush=True)




###########################################################
# 3. Batch : Multiple Inputs at once (runs in Parallels)
###########################################################
responses = llm.batch([
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
])

for index, Response in enumerate(responses):
    print(f"Response for {index+1} : {Response.content}")
    print("--------------------------------------------")




###########################################################
# 4 .ainvoke() — Async version (for web apps)
###########################################################
async def async_example():
    response = await llm.ainvoke("What is AI?")
    print("Async response:", response.content)

asyncio.run(async_example())




###########################################################
# 5 Async streaming 
###########################################################
async def async_stream_example():
    async for chunk in llm.astream("Tell me a joke"):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(async_stream_example())