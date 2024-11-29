from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import openai
import logging
# logging.basicConfig(level=logging.DEBUG)

# Set up the environment variables or values for API key and URL
OPENAI_BASE_URL = 'https://llm.ai.broadcom.net/api/v1'
OPENAI_API_KEY = "57fa6c09-20a4-4cc0-892e-23d0a37b26c2"

# Initialize the OpenAIEmbeddings from LangChain
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL,
    model="BAAI/bge-en-icl",
    tiktoken_enabled=False,
    # encoding_format="float",
    model_kwargs={"encoding_format": "float"}
)

# Create embeddings for a given input
response = embeddings.embed_documents(["today is monday"])
print(response)

