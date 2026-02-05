import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq # Package not installed
load_dotenv()
from pydantic import BaseModel,Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage

from tools import *
if "GROQ_API_KEY" not in os.environ:
    print("Warning: GROQ_API_KEY not found in environment variables.")

class TravelPlan(BaseModel):
    destination: str
    duration: int
    budget: int
    activities: List[str]
    accommodations: List[str]
    transportation: List[str]
    safety: List[str]
    health: List[str]
    culture: List[str]
    itinerary: List[str]
    
tools = [get_location_by_ip]
# Using ChatOpenAI with Groq endpoint
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="qwen/qwen3-32b" # Valid Groq model'
   
).bind_tools(tool for tool in tools)

# Test run function
def llm_node(state: dict):
    messages: list[BaseMessage] = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}



