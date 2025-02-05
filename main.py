from fastapi import FastAPI
from pydantic import BaseModel 
from chatbot import generate_response

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
async def search(request: QueryRequest):
    response = generate_response(request.query)
    return {"response": response.content}