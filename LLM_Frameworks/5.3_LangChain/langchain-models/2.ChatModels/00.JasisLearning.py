from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temprature=0.5

)

memory = []

while True:
    print("----------------- YOU -------------------")
    prompt = input("Enter your Prompt: ")
    memory.append(prompt)

    if prompt == "exit":
        break

    response = llm.invoke(memory)
    memory.append(response.content)
    print("----------------- AI -------------------")
    print(response.content)

for i in memory:
    print(f"## -> {i}")




