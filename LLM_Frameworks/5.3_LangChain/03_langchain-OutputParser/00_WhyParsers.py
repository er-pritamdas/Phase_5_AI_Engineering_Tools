# ============================================================
# THE PROBLEM — LLMs always return raw text
# ============================================================

# What the LLM gives you:
# "The movie Inception has a rating of 9.2 out of 10. It's a mind-bending thriller."

# What your app actually needs:
# {"title": "Inception", "rating": 9.2, "summary": "A mind-bending thriller"}

# You can't put a paragraph into a database.
# You can't use a sentence as an API response.
# Output Parsers convert raw text → structured Python data.


from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.5
)


# ============================================================
# 1. StrOutputParser — Just Give Me the Text
#    Strips AIMessage wrapper, returns plain string
# ============================================================

# from langchain_core.output_parsers import StrOutputParser

# parser = StrOutputParser()
# response = llm.invoke("What is Python")
# output = parser.invoke(response)
# print(output)



# ============================================================
# 2. JsonOutputParser — Give Me a Python Dictionary
#    Forces LLM to respond in JSON, parses into dict
# ============================================================


from langchain_core.output_parsers import JsonOutputParser

JsonParser = JsonOutputParser()

# The parser can generate instructions telling the LLM how to format its response
print("\nJSON format instructions:")
print(JsonParser.get_format_instructions()) # O/P : "Return a JSON Object"

template = ChatPromptTemplate.from_messages([
    ("system", "Respond ONLY in JSON format."),
    ("user", "Give me about the country : {country}"
              "Include Name, Capital, Population and Continent."
              "{Instruction}")
]
)

prompt= template.invoke({
    "country": "India",
    "Instruction" : JsonParser.get_format_instructions()
})

response = llm.invoke(prompt)
print(response.content)