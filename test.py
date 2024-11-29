import openai
import logging
logging.basicConfig(level=logging.DEBUG)
OPENAI_BASE_URL = 'https://llm.ai.broadcom.net/api/v1'
OPENAI_API_KEY = "57fa6c09-20a4-4cc0-892e-23d0a37b26c2"
client=openai.OpenAI(
  base_url=OPENAI_BASE_URL,
  api_key=OPENAI_API_KEY
)

response=client.embeddings.create(
  model="BAAI/bge-en-icl",
  input=[
  "today is monday"
],
  encoding_format="float"
)
print(response)
