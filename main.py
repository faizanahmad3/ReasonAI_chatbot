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

openai_client = OpenAI()
qdrant_client = QdrantClient(
    url="https://c79cd5b9-7bef-4139-979b-998b85f7fb26.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="YBfdnGJ7XHg4nHtz9eR-Fbrlpys6uGp8-DBYIlMNekDzPExey3fu0w",
)