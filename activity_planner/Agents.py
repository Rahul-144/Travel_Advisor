from langgraph.graph import END, StateGraph
# from langgraph.prompts import ChatPromptTemplate
import operator
from typing import List, Dict, TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage,ToolMessage
from Model import llm_node
from Faiss_indexing import faiss_index
from tools import get_location_by_ip

TOOLS = {
    "get_location_by_ip": get_location_by_ip
}
class AgentState(TypedDict):
    messages:Annotated[List[BaseMessage], operator.add]
class Agent():
    def __init__(self, system=""):
        self.system= system
        self.tools = TOOLS
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
        self.graph = graph.compile()
  
    def Rag_node(self,state: AgentState):
        query = state["messages"][-1].content
        vectorstore = faiss_index()
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(query)

        context = "\n".join(d.page_content for d in docs)

        messages = []
        if self.system:
            messages.append(SystemMessage(content=self.system))

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
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
    def run(self, query: str):
        return self.graph.invoke(
            {"messages": [HumanMessage(content=query)]}
        )
prompt = """You are a smart travel research assistant don't answer any unrelated question.
Answer only using the given context. Use tools for customizing the response.
You MUST respond in valid JSON ONLY.
The JSON must match this schema when it is asked to plan a trip:

{
  "destination": string,
  "duration": number,
  "budget": number,
  "activities": string,
  "accommodations": string,
  "transportation": string,
  "safety": string,
  "health": string,
  "culture": string,
  "itinerary": string
}

Rules:
- Use ONLY the provided context
- If information is missing, use empty arrays or "unknown"
- Do NOT add explanations or markdown"""
abot = Agent(system=prompt)
while True:
    query = input("Enter your query: ")
    if query.lower() == "exit":
        break
    result = abot.run(query)
    print(result["messages"][-1].content)
    if query.lower() == "exit":
        break