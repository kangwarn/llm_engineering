#!/usr/bin/env python
# coding: utf-8

# In[15]:


from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import gradio as gr
from dotenv import load_dotenv
import os
from langchain_core.callbacks import StdOutCallbackHandler


# In[20]:


# Initialize OpenAI LLM
load_dotenv()
os.environ['OPENAI_API_KEY'] = "57fa6c09-20a4-4cc0-892e-23d0a37b26c2"
llm = ChatOpenAI(temperature=0.7, model="meta-llama/Meta-Llama-3-70B-Instruct", base_url="https://llm.ai.broadcom.net/api/v1")

# Create a conversation chain with memory
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, callbacks=[StdOutCallbackHandler()])

# Generator function for streaming chatbot responses
def chat_with_bot_stream(user_message, history):
    # Add user's message to the memory
    memory.chat_memory.add_user_message(user_message)
    
    # Start streaming response
    response = conversation.llm._call(user_message, stop=None)  # Directly call the LLM's method for streaming
    bot_reply = ""
    for chunk in response.split("\n"):
        bot_reply += chunk
        # Append to chat history and yield the intermediate output
        history.append((user_message, bot_reply))
        yield history, history

def chat(question, history):
    result = conversation.invoke({"input": question})
    # return result["response"]
    history.append((question, result["response"]))
    return history, history

def chat_with_bot(user_message, history):
    # Add user's message to the memory
    memory.chat_memory.add_user_message(user_message)
    
    # Get the chatbot's reply
    bot_reply = conversation.run(user_message)
    
    # Append both user message and bot reply to the chat history
    history.append((user_message, bot_reply))
    
    return history, history

# Gradio Interface
with gr.Blocks() as gr_interface:
    chatbot = gr.Chatbot(label="Chat with LangChain + OpenAI (Streaming)")
    msg = gr.Textbox(placeholder="Type your message here...")
    clear_btn = gr.Button("Clear")
    
    # Initialize chat history
    state = gr.State([])

    # Define interaction
    msg.submit(chat, [msg, state], [chatbot, state])
    clear_btn.click(lambda: ([], []), None, [chatbot, state])

# Run the Gradio app
gr_interface.launch()


# In[ ]:




