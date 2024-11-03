
# This code is the demonstrate a simple way of forming a Prompt and using it to Chain with a Model  

'''
**Step-by-Step Explanation of the Code**

### 1. Importing Required Libraries
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
```
- **`ChatPromptTemplate`**: This class is used to create a **template for prompts** that can later be filled with variables (like `{name}` in this case).
- **`ChatOpenAI`**: Used to connect to **OpenAI’s chat models** like `gpt-3.5-turbo`.
- **`ChatGoogleGenerativeAI`**: Used to connect to **Google’s Generative AI (Gemini) models**.
- **`load_dotenv()`**: Loads environment variables (like API keys) from a `.env` file.
- **`os`**: Standard Python module to interact with environment variables.

### 2. Load Environment Variables
```python
load_dotenv()
```
- This line ensures that the environment variables (like `OPENAI_API_KEY` or `GOOGLE_API_KEY`) are loaded from a `.env` file. These keys are necessary to authenticate with OpenAI or Google Generative AI.

### 3. Define `main()` and `demosimple()` Functions
```python
def main():
    print(demosimple.__doc__)  # Print the docstring of demosimple function
    demosimple()  # Call the demosimple function
```
- The `main()` function is a standard entry point to run the program.
- It prints the **docstring** of the `demosimple()` function, which explains the function's purpose.
- Then, it **calls** the `demosimple()` function to demonstrate the LangChain expression logic.

### 4. Define the `demosimple()` Function
```python
def demosimple():
    """
    This Function Demonstrates a simple use of LCEL (LangChain Expression Language) to create a custom Chain with the Prompt and Model
    """
```
- This docstring explains that the `demosimple()` function demonstrates **LCEL** (LangChain Expression Language), which allows chaining models with prompts.

### 5. Create the Prompt Template
```python
prompt = ChatPromptTemplate.from_template("Tell me a few key achievements of {name}")
```
- **`ChatPromptTemplate.from_template()`**:
  - This method creates a **template prompt** that includes a **placeholder variable** `{name}`.
  - Later, you’ll **fill in the value** for `{name}` (in this case, `"Abraham Lincoln"`).
  - The final prompt might look like: **"Tell me a few key achievements of Abraham Lincoln"**.

### 6. Initialize the LLM Model
```python
# Option 1: Use OpenAI GPT-3.5
# model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)

# Option 2: Use Google's Gemini Pro
model = ChatGoogleGenerativeAI(
    model="gemini-pro",  
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5
)
```
- This section shows **two options** for initializing the **LLM**:
  1. **OpenAI’s GPT-3.5**.
  2. **Google’s Gemini Pro (Generative AI)**.
- The **model parameters**:
  - **`model_name`**: Specifies which model to use.
  - **`api_key`**: The API key is **retrieved from environment variables** using `os.getenv()`.
  - **`temperature`**: Controls the **creativity** of the model’s responses.
    - **Low temperature** (e.g., 0.5) makes responses more **focused** and **deterministic**.
    - **Higher temperature** makes responses more **diverse** and **creative**.

### 7. Create the Chain Using LCEL (LangChain Expression Language)
```python
chain = prompt | model  # LCEL - LangChain Expression Language
```
- **LCEL (LangChain Expression Language)** allows **"chaining"** the prompt with the model using the **`|` operator**.
- This chain means:
  1. **The prompt** is filled with the user input (e.g., `"Abraham Lincoln"`).
  2. **The model** processes the prompt to generate a response.

### 8. Invoke the Chain
```python
print(chain.invoke({"name": "Abraham Lincoln"}).content)
```
- **`chain.invoke()`**: This method **runs the chain** by:
  1. **Filling the template** with the input value for `{name}` (in this case, `"Abraham Lincoln"`).
  2. **Generating a response** from the chosen model.
- **Example Input**:
  - The input dictionary is `{"name": "Abraham Lincoln"}`.
- **Example Output**:
  - The prompt becomes: **"Tell me a few key achievements of Abraham Lincoln"**.
  - The model (e.g., Gemini or GPT-3.5) generates a response like:
    ```
    "Abraham Lincoln led the United States through the Civil War, preserved the Union, abolished slavery, and delivered the Gettysburg Address."
    ```

### 9. Full Code Summary
Here’s how the code works from start to finish:
1. **Load API keys** using `load_dotenv()`.
2. **Create a prompt template** with `{name}` as a placeholder.
3. **Initialize the model** (OpenAI GPT or Google Gemini).
4. **Chain the prompt with the model** using **LCEL**.
5. **Invoke the chain** with an input value (`"Abraham Lincoln"`).
6. **Print the generated response** from the model.

### Complete Code for Reference
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print(demosimple.__doc__)
    demosimple()

def demosimple():
    """
    This Function Demonstrates a simple use of LCEL (LangChain Expression Language) to create a custom Chain with the Prompt and Model.
    """

    # Create the Prompt Template
    prompt = ChatPromptTemplate.from_template("Tell me a few key achievements of {name}")

    # Create the LLM Object (options between OpenAI GPT or Gemini)
    # model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.5
    )

    # Create the Chain using LCEL
    chain = prompt | model

    # Invoke the Chain with input
    print(chain.invoke({"name": "Abraham Lincoln"}).content)

if __name__ == "__main__":
    main()
```

### Key Takeaways
- **LangChain Expression Language (LCEL)** allows seamless chaining of prompts and models with minimal code.
- **Prompt templates** make it easy to create reusable prompts with variables.
- **Chaining** simplifies complex NLP workflows by connecting components in a readable way.
- **Google Gemini** and **OpenAI GPT models** can be swapped easily using LangChain’s interface.

This code demonstrates how to leverage LangChain’s **flexible expression language** to build simple, maintainable workflows for interacting with LLMs.


Full Code Summary

Here’s how the code works from start to finish:

Load API keys using load_dotenv().

Create a prompt template with {name} as a placeholder.

Initialize the model (OpenAI GPT or Google Gemini).

Chain the prompt with the model using LCEL.

Invoke the chain with an input value ("Abraham Lincoln").

Print the generated response from the model.

'''
import sys

# Print the path to the Python interpreter being used
print(sys.executable)


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import HuggingFaceHub
#from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os 


load_dotenv()

def main():

    print(demosimple.__doc__)
    demosimple() 

    print(demosimple_with_huggingface.__doc__)
    demosimple_with_huggingface()
         
def demosimple():
    """
    This Function Demonstrates a simple use of LCEL (LangChain Expression Language) to create a custom Chain with the Prompt and Model
    """

    # Create the Prompt Template
    #prompt = ChatPromptTemplate.from_template("Tell me a few key achievements of {name}")
    #prompt = ChatPromptTemplate.from_template("What is the capital name of  {name}")
    prompt = ChatPromptTemplate.from_template("What is the signs of  {name}")
    # Create the LLM Object (options between OpenAI GPT or Gemini)
    #model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)
    model = ChatGoogleGenerativeAI(model="gemini-pro",  google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.8)

    # Create the Chain
    chain = prompt | model     # LCEL - LangChain Expression Language
    
    # Invoke (run) the Chain - The Chat Model returns a Message
    #print(chain.invoke({"name": "Abraham Lincoln"}).content)
    #print(chain.invoke({"name": "United Arab Emirates"}).content)
    print(chain.invoke({"name": "babesia in horse"}).content)


def demosimple_with_huggingface():
    """
    This Function Demonstrates a simple use of LCEL (LangChain Expression Language) 
    to create a custom Chain with the Prompt and Model using Hugging Face Hub.
    """

    # Create the Prompt Template
    
    #prompt = ChatPromptTemplate.from_template("What is the capital name of {name}")
    
    prompt = ChatPromptTemplate.from_template("What is your name {param}")
    
    # Create the Hugging Face Hub LLM Object
    # Initialize the Hugging Face Hub model via LangChain
    #model_name="google/flan-t5-base"
    model = HuggingFaceHub( 
        repo_id="openai-community/gpt2-large",  # You can replace 'gpt2' with a different model available on Hugging Face Hub
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_kwargs={"temperature": 0.5, "max_length": 100}
    )

    # Create the Chain using LCEL
    chain = prompt | model
    
    # Invoke (run) the Chain - The Chat Model returns a Message
    #print(chain.invoke({"name": "United Arab Emirates"}))
    print(chain.invoke({"param": " and what is your profession"}))

if __name__ == "__main__":
    main()
    
