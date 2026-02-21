from fastapi import FastAPI
from .Agents import get_agent
from pydantic import BaseModel

app = FastAPI()
agent = get_agent()

# optional memory thread
class Query(BaseModel):
    question: str
    thread_id: str = "1" 
@app.post("/ask")
async def ask(query: Query):

    thread = {"configurable": {"thread_id": query.thread_id}}

    result = agent.run(query.question, thread=thread)

    return {
        "answer": result["messages"][-1].content
    }
