import streamlit as st
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A Chatbot With OPENAI"

## Prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):
    llm=ChatGroq(model=llm,groq_api_key=api_key)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

## Streamlit app
st.title("Enhanced Q&A Chatbot with OpenAI")

## Sidebar for the settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

## Dropdown to select various Groq models
llm=st.sidebar.selectbox("Select a Groq Model",["llama3-70b-8192","llama3-8b-8192","gemma2-9b-it"])

## Adjust response parameter 
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input=st.text_input("You: ")

if user_input and api_key:
    response=generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
    
elif user_input:
    st.warning("Please enter the Groq Api key in the side bar")    

else:
    st.write("Please provide the query")