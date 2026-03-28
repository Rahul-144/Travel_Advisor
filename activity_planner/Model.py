import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq # Package not installed
load_dotenv()
from pydantic import BaseModel,Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage

from tools import *
if "OPEN_API_KEY" not in os.environ:
    print("Warning: OPEN_API_KEY not found in environment variables.")

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
    
tools = [get_current_date,get_location_by_ip, search_flights, search_hotels]
# Using ChatOpenAI with Groq endpoint
llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.environ.get("OPEN_API_KEY"),
    model_name="qwen/qwen3-32b" # Valid Groq model'
   
).bind_tools(tools)

# Test run function
def llm_node(state: dict):
    messages: list[BaseMessage] = state["messages"]

    # Prepend system prompt if present in state (avoids needing a separate node)
    system_prompt = state.get("system_prompt", "")
    if system_prompt:
        messages = [SystemMessage(content=system_prompt)] + list(messages)

    response = llm.invoke(messages)

    # Qwen3 (and other thinking models) wrap chain-of-thought in <think>...</think>.
    # Strip it out so the downstream JSON parser always gets clean output.
    import re
    clean_content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
    response.content = clean_content

    result = {"messages": [response]}
    # pass citation along if present
    if "citation" in state:
        result["citation"] = state["citation"]
    return result


        