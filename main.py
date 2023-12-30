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

with open("path.yml", "r") as p:
    config = yaml.safe_load(p)

openai_client = OpenAI()
qdrant_client = QdrantClient(
    url=config['QDRANT_CLUSTER_URL'],
    api_key=qdrant_api_key,
)