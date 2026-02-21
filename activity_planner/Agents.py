from langgraph.graph import END, StateGraph
# from langgraph.prompts import ChatPromptTemplate
import operator
from typing import List, Dict, TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage,ToolMessage, AnyMessage
from uuid import uuid4
# from langgraph.graph.message import add_messages
from .Model import llm_node
from .Faiss_indexing import faiss_index
from .tools import *

from langgraph.checkpoint.memory import MemorySaver
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=3)

import json

TOOLS = {
    "get_location_by_ip": get_location_by_ip,
    "search_flights": search_flights,
    "search_hotels": search_hotels,
    "get_current_date": get_current_date
    }
# memory = SqliteSaver.from_conn_string(":memory:")
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged
class AgentState(TypedDict):
    messages:Annotated[List[BaseMessage], reduce_messages]
    # messages: Annotated[List[BaseMessage], add_messages]
class Agent():
    def __init__(self, system=""):
        self.system= system
        self.flags = True
        self.tools = TOOLS
        self.vectorstore = faiss_index()
        graph = StateGraph(AgentState)
        graph.add_node("llm",llm_node)
        graph.add_node("rag",self.Rag_node)
        graph.add_node("action",self.take_action)
        
        graph.add_edge("rag","llm")
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("rag")
        self.memory = MemorySaver()
        
        # 3. Compile with the persistent saver
        self.graph = graph.compile(checkpointer=self.memory)
  
    def Rag_node(self,state: AgentState):
        query = state["messages"][-1].content
       
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

        docs = docs = compression_retriever.invoke(query)

        context = "\n".join(d.page_content for d in docs)

        messages = []
        if self.system and self.flags:
            messages.append(SystemMessage(content=self.system))
            self.flags = False
        messages.append(SystemMessage(content=f"Context:\n{context}"))
        messages.append(HumanMessage(content=query))
        return {"messages": messages}
    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t['name'] not in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=json.dumps(result)))
        print("Back to the model!")
        return {'messages': results}
    def run(self, query: str, thread: Dict = None):
        initial_messages = []
        # if self.system:
        #     initial_messages.append(SystemMessage(content=self.system))
        initial_messages.append(HumanMessage(content=query))

        return self.graph.invoke(
            {"messages": initial_messages},
            config=thread
        )
prompt =''' You are a smart and friendly travel research assistant.

You MUST follow these rules strictly:

GENERAL RULES:
- Answer using the provided context.
- Try to get as much information as possible from the context.
- **CRITICAL**: If the context is missing information (e.g., specific hotels, flight prices, current events) or if the user asks for real-time data, YOU MUST USE THE AVAILABLE TOOLS (serpapi_search, search_flights, etc.).
- Do NOT use internal training knowledge not present in the context or tool outputs.
- Do NOT infer or assume missing information.
- You MUST always respond in valid JSON.
- Do NOT add explanations, markdown, or extra text.

INTENT HANDLING:
1. If the user greets (e.g., "hi", "hello", "hey") or asks something unrelated to travel planning:
   Respond with this JSON schema ONLY:

   {
     "type": "non_trip",
     "message": string
   }

2. If the user asks to plan a trip or requests travel-related information:
   Respond using the following schema ONLY:

   {
     "type": "trip_plan",
     "destination": string,
     "duration_days": number | "unknown",
     "budget_estimate": number | "unknown",
     "activities": string[],
     "accommodations": string[],
     "transportation": string[],
     "safety": string[],
     "health": string[],
     "culture": string[],
     "itinerary": string[],
     "tools_used": string[]
   }

CONTEXT RULES:
- Every field must be grounded in the provided context or tool results.
- If a field cannot be answered from the context/tools, return an empty array or "unknown".
- If you used a tool, list it in "tools_used".

Violating any rule is considered an error.
'''

_agent_instance = None
def get_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = Agent(system=prompt)
    return _agent_instance