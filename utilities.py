import json
import openai
import qdrant_client.http.models
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.messages import BaseMessage
# from jose import jwt
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, FilterSelector
from qdrant_client.http.models import Batch
from qdrant_client.http import models as rest
import os
import uuid
import csv
from datetime import datetime


def loading_docs(path):
    if path.endswith('.pdf'):
        loader = PyPDFLoader(path)
        return loader.load()
    elif path.endswith('.docx'):
        loader = Docx2txtLoader(path)
        return loader.load()
    else:
        print("not a valid file")


def split_text(pages, chunksize=700, chunkoverlap=70):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunksize,
        chunk_overlap=chunkoverlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(pages)
    return docs


def qdrant_instance_func(db_client, embeddings):
    return qdrant.Qdrant(
        client=db_client,
        collection_name="user1",
        embeddings=embeddings
    )


def create_embeddings(chunks, db_client, embeddings):
    print("creating embeddings with Qdrant")
    qdrant_instance = qdrant_instance_func(db_client, embeddings)
    qdrant_instance.from_documents(documents=chunks, embedding=embeddings, collection_name="user1")
