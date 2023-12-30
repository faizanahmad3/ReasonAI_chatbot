import json
import openai
import qdrant_client.http.models
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
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

def split_text(pages, chunksize, chunkoverlap):
    """

    Args:
      pages:
      chunksize:
      chunkoverlap:

    Returns:

    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunksize,
        chunk_overlap=chunkoverlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(pages)
    return docs

def create_embeddings(pdf_path):
    pdf = os.listdir(pdf_path)
