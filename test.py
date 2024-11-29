import os 
from dotenv import load_dotenv
# Load environment variables
load_dotenv()



api_key = os.getenv("Langchain_project_openai")
print(os.getenv("Langchain_project_openai"))


# from transformers import pipeline
# from langchain.llms import HuggingFacePipeline

# # Initialize a Hugging Face pipeline for text generation
# generator = pipeline("text-generation", model="gpt2")
# llm = HuggingFacePipeline(pipeline=generator)

# # Generate text using the pipeline
# response = llm("Explain the concept of democracy.")
# print(response)


# from transformers.utils.hub import TRANSFORMERS_CACHE

# print(TRANSFORMERS_CACHE)  # This shows the current cache directory



# Use the API to send prompts and receive responses
# import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")

# response = openai.chat.completions.create(
#     model="o1-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Explain Python decorators"}
#     ]
# )

# print(response['choices'][0]['message']['content'])

# # To programmatically list available models:

# import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")

# models = openai.models.list()
# for model in models:
#     print(model)



# anthropic

import anthropic

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.getenv("anthropic_api_key"),

)

message = client.messages.create(
    model="new-model-name",  # Change the model name here
    max_tokens=1000,
    temperature=0,
    system="what you know about langchain\n",
    messages=[]
)
print(message.content)

