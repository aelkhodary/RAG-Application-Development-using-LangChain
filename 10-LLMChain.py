# This program demonstrates a simple usage of creating a LCEL based chain 
# The chain comprises a prompt, the llm object and a Stringoutput parser

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import HuggingFaceHub
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()


#llm = ChatOpenAI()

# Create the Hugging Face Hub LLM Object
# Hugging Face Hub LLM https://huggingface.co/api/models
llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-7B-Chat",  # You can replace 'gpt2' with another model of your choice
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_kwargs={"temperature": 0.7, "max_length": 50}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

output = chain.invoke({"input": "how can langsmith help with testing?"})

print(output)



