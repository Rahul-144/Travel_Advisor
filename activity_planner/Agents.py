from langgraph.graph import END, StateGraph
# from langgraph.prompts import ChatPromptTemplate
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import operator
from typing import List, Dict, TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage,ToolMessage, AnyMessage
from uuid import uuid4
# from langgraph.graph.message import add_messages
from Model import llm_node
from Faiss_indexing import faiss_index
from tools import *

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
    citation:  Dict[str, str]
    content:  Dict[str, str]
    question_type: str
    system_prompt: str
class Agent():
    def __init__(self, system=""):
        self.system = system
        self.tools = TOOLS
        self.vectorstore = faiss_index()
        graph = StateGraph(AgentState)
        graph.add_node("classify", self.classify_node)
        graph.add_node("llm",llm_node)
        graph.add_node("rag",self.Rag_node)
        graph.add_node("action",self.take_action)
        
        # Classifier routes to either RAG (non-trip) or directly to LLM (trip)
        graph.add_conditional_edges(
            "classify",
            self.route_by_question_type,
            {"non_trip": "llm", "trip": "rag"}
        )
        graph.add_edge("rag","llm")
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("classify")
        self.memory = MemorySaver()
        
        # Compile with the persistent saver
        self.graph = graph.compile(checkpointer=self.memory)
  
    def classify_node(self, state: AgentState):
        """Classify whether the question is trip-related or not."""
        query = state["messages"][-1].content

        # System prompt for classification
        classifier_prompt = """You are a travel question classifier. Classify the user's question into one of two categories:

1. "trip" - If the question is about travel planning, trips, destinations, accommodations, flights, activities, or any travel-related topic
2. "non_trip" - If the question is a greeting, general knowledge question, or anything not related to travel

Respond with ONLY the category name, nothing else."""

        messages = [
            SystemMessage(content=classifier_prompt),
            HumanMessage(content=query)
        ]

        # Use a lightweight LLM for classification
        from Model import llm
        response = llm.invoke(messages)
        raw = response.content.strip().lower()

        # Validate — guard against empty string (tool-call response) or verbose output
        if raw.startswith("trip"):
            question_type = "trip"
        else:
            question_type = "non_trip"  # default: treat anything unclear as travel-related

        print(f"[CLASSIFIER] Query: {query} → Type: {question_type}")

        return {
            "question_type": question_type,
            "messages": state["messages"]
        }
    
    def route_by_question_type(self, state: AgentState):
        return state["question_type"]
  
    def Rag_node(self,state: AgentState):
        query = state["messages"][-1].content
       
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

        docs = compression_retriever.invoke(query)

        context = "\n".join(d.page_content for d in docs)
        
        # Extract citations from retrieved documents
        citations = []
        contents = []
        for doc in docs:
            metadata = doc.metadata
            print(f"Metadata for retrieved doc: {metadata}")  # Debug print
            citation = {
                "source": metadata.get("source", "Unknown"),
                "seq_num": metadata.get("seq_num", "Unknown")
            }
            content = {"context": doc.page_content}
            if content not in contents:
                contents.append(content)  # avoid duplicates
            if citation not in citations:  # avoid duplicates
                citations.append(citation)
        citation_str = json.dumps(citations)
        content_str = json.dumps(contents)

        messages = []
        messages.append(SystemMessage(content=f"Context:\n{context}"))
        messages.append(HumanMessage(content=query))
        return {"messages": messages, "citation": citation_str, "content": content_str}
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
        ret = {'messages': results}
        if 'citation' in state:
            ret['citation'] = state['citation']
        return ret
    def run(self, query: str, thread: Dict = None):
        initial_messages = [HumanMessage(content=query)]
        return self.graph.invoke(
            {
                "messages": initial_messages,
                "citation": "",
                "content": "",
                "question_type": "",
                "system_prompt": self.system
            },
            config=thread
        )
prompt =''' You are a smart and friendly travel research assistant.

You MUST follow these rules strictly:

GENERAL RULES:
- When ever use of date comes up, use the "get_current_date" tool to get today's date. Do NOT rely on any internal clock or assumptions about the date.
- Answer using the provided context.
- Try to get as much information as possible from the context.
- Do not answer any knowledge questions that are not related to travel or grounded in the context or tool outputs.
- **CRITICAL**: If the context is missing information (e.g., specific hotels, flight prices, current events) or if the user asks for real-time data, YOU MUST USE THE AVAILABLE TOOLS (serpapi_search, search_flights, etc.).
- Do NOT infer or assume missing information.
- You MUST always respond in valid JSON.
- Do NOT add explanations, markdown, or extra text.

INTENT HANDLING:
1. If the user greets (e.g., "hi", "hello", "hey"):
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
     "flights": [{"airline": string, "flight_number": string, "departure_time": string, "arrival_time": string, "duration_minutes": number, "price_inr": number}],
     "hotels": [{"name": string, "rating": number, "price_per_night": string, "total_price": string, "amenities": string[]}],
     "transportation": string[],
     "safety": string[],
     "health": string[],
     "culture": string[],
     "itinerary": string[],
     "tools_used": string[]
   }

   - If you called the "search_flights" tool, populate the "flights" array with the exact data returned.
   - If you called the "search_hotels" tool, populate the "hotels" array with the exact data returned.
   - If no tool was called for flights/hotels, return an empty array [].

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
