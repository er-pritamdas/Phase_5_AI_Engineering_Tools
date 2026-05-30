from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="gemma3:latest",
    base_url="http://localhost:11434",
    temperature=0.5
)


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

# prompt= template.invoke({
#     "country": "India",
#     "Instruction" : JsonParser.get_format_instructions()
# })

# response = llm.invoke(prompt)
# print(response.content)


# =============================== Using Chains: ==================================

chain = template | llm | JsonParser
response = chain.invoke({
    "country": "India",
    "Instruction" : JsonParser.get_format_instructions()
})
print(response)
print(response["name"])
print(response["capital"])





# ============================================================
# 3. PydanticOutputParser — Give Me a Validated Python Object
#    Most powerful: define a class, get typed + validated data
# ============================================================
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- Step 1: Define the shape you want ---
# This is a Pydantic model — a Python class with typed fields
class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating out of 10")
    pros: list[str] = Field(description="List of good things about the movie")
    cons: list[str] = Field(description="List of bad things about the movie")
    recommend: bool = Field(description="Would you recommend this movie?")

# --- Step 2: Create the parser ---
pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)

# See what instructions it generates for the LLM
print("\nPydantic format instructions:")
print(pydantic_parser.get_format_instructions())

# --- Step 3: Build prompt with format instructions ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic. {format_instructions}"),
    ("user", "Review the movie: {movie}")
])

chain = prompt | llm | pydantic_parser

# --- Step 4: Use it! ---
result = chain.invoke({
    "movie": "Inception",
    "format_instructions": pydantic_parser.get_format_instructions()
})

print("\nPydantic Parser result:")
print(type(result))        # <class 'MovieReview'>  ← Python object!
print(result.title)        # "Inception"
print(result.rating)       # 9.1
print(result.pros)         # ["Mind-bending plot", "Great visuals"]
print(result.cons)         # ["Can be confusing"]
print(result.recommend)    # True

# Why Pydantic is powerful:
# - Dot access: result.title instead of result["title"]
# - Types are VALIDATED: rating must be float, not string
# - IDE autocomplete works on the fields
# - Converts to JSON automatically for APIs




# ============================================================
# 4. .with_structured_output() — The Modern & Cleanest Way
#    No separate parser needed! Attach directly to the model.
#    Uses function calling under the hood — most reliable.
# ============================================================

# --- Same MovieReview, but much cleaner ---
class MovieReview2(BaseModel):
    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="One sentence summary")
    recommend: bool = Field(description="Would you recommend it?")

# Attach structured output directly to the model
structured_llm = llm(model="gpt-4o").with_structured_output(MovieReview2)

# No format_instructions, no separate parser, no prompt hacking!
result = structured_llm.invoke("Review the movie Interstellar")

print("\nStructured output result:")
print(type(result))           # <class 'MovieReview2'>
print(f"Title: {result.title}")         # "Interstellar"
print(f"Rating: {result.rating}")       # 9.3
print(f"Summary: {result.summary}")     # "A visually stunning..."
print(f"Recommend: {result.recommend}") # True


# --- Using structured output in a chain ---
class CountryInfo(BaseModel):
    name: str = Field(description="Country name")
    capital: str = Field(description="Capital city")
    population: int = Field(description="Approximate population")
    languages: list[str] = Field(description="Official languages")
    fun_fact: str = Field(description="An interesting fact")

structured_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(CountryInfo)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a geography expert."),
    ("user", "Tell me about {country}")
])

# Note: structured_llm already handles parsing, no StrOutputParser needed
chain = prompt | structured_llm

result = chain.invoke({"country": "Brazil"})

print(f"\nCountry: {result.name}")
print(f"Capital: {result.capital}")
print(f"Population: {result.population}")
print(f"Languages: {result.languages}")
print(f"Fun fact: {result.fun_fact}")


# ============================================================
# 5. COMPARISON — All 4 approaches side by side
# ============================================================

# --- Approach 1: StrOutputParser ---
# chain = prompt | llm | StrOutputParser()
# result = chain.invoke(...)  → "plain text string"

# --- Approach 2: JsonOutputParser ---
# chain = prompt_with_format_instructions | llm | JsonOutputParser()
# result = chain.invoke(...)  → {"key": "value"}  (dict)

# --- Approach 3: PydanticOutputParser ---
# chain = prompt_with_format_instructions | llm | PydanticOutputParser(pydantic_object=MyModel)
# result = chain.invoke(...)  → MyModel(field=value)  (Pydantic object)

# --- Approach 4: .with_structured_output() ---
# structured_llm = llm.with_structured_output(MyModel)
# chain = prompt | structured_llm
# result = chain.invoke(...)  → MyModel(field=value)  (cleanest!)


# ============================================================
# SUMMARY CHEAT SHEET
# ============================================================
#
# | Parser                      | Returns       | Reliability | Use When                    |
# |-----------------------------|---------------|-------------|-----------------------------|
# | StrOutputParser             | str           | 100%        | You just want plain text    |
# | JsonOutputParser            | dict          | Good        | Need dict, no validation    |
# | PydanticOutputParser        | Pydantic obj  | Good        | Need typed, validated data  |
# | .with_structured_output()   | Pydantic obj  | Best        | Modern, cleanest approach   |
#
# RECOMMENDATION:
# - Text output → StrOutputParser
# - Structured output → .with_structured_output()
# - Skip JsonOutputParser and PydanticOutputParser unless you have a specific reason
#
# ============================================================
