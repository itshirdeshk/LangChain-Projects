from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os

from dotenv import load_dotenv

load_dotenv("./.env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(GEMINI_API_KEY)

app = FastAPI(
    title="LangChain Server", version="1.0", description="A Simple API Server"
)

add_routes(
    app,
    ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY),
    path="/gemini",
)
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

prompt1 = ChatPromptTemplate.from_template(
    "Write an essay about {topic} with 100 words."
)
prompt2 = ChatPromptTemplate.from_template(
    "Write an poem about {topic} with 100 words."
)

add_routes(app, prompt1 | model, path="/essay")
add_routes(app, prompt2 | model, path="/poem")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
