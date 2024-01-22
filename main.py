from utilities import loading_docs, split_text, create_embeddings, qdrant_instance_func
from LLM_models import embedding_model
import os
import yaml
import uuid
from bson import ObjectId
# from jose import JWTError, jwt
from openai import OpenAI
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import Batch
# from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

qdrant_api_key = os.getenv("QDRANT_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
OpenAIEmbeddings.openai_api_key = os.getenv("OPENAI_API_KEY")
llm = embedding_model

with open("path.yml", "r") as p:
    config = yaml.safe_load(p)

openai_client = OpenAI()
db_client = QdrantClient("localhost", port=6333)

document_path = "demo_files/langchain vs llama-index.docx"
pages = loading_docs(path=document_path)
chunks = split_text(pages)
print("chunks created")
create_embeddings(chunks, db_client, llm)
