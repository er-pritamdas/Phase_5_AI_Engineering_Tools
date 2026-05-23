from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
import warnings

warnings.filterwarnings("ignore")

os.environ['HF_HOME'] = '/home/pritam/Pulse/LangChain/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    device=0,   # GPU
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100,
        "return_full_text": False,
        "clean_up_tokenization_spaces": False
    }
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the Height of a Normal Human Being")

print(result.content)