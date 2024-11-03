import os 
api_key = os.getenv("Langchain_project_openai")
print(os.getenv("Langchain_project_openai"))


from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Initialize a Hugging Face pipeline for text generation
generator = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=generator)

# Generate text using the pipeline
response = llm("Explain the concept of democracy.")
print(response)


from transformers.utils.hub import TRANSFORMERS_CACHE

print(TRANSFORMERS_CACHE)  # This shows the current cache directory
