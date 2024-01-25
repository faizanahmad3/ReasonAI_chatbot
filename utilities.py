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


# user_id = "user1"


def iterating_over_docs(source_directory, user_id):
    root_dir = str(os.path.join(source_directory, user_id))
    file_path = [os.path.join(root_dir, single_file) for single_file in os.listdir(root_dir)]
    return file_path


def loading_docs(path_list):
    docs = []
    for path in path_list:
        if path.endswith('.pdf'):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif path.endswith('.docx'):
            loader = Docx2txtLoader(path)
            docs.extend(loader.load())
        else:
            print("not a valid file")
    return docs


def split_text(pages, chunksize=700, chunkoverlap=70):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunksize,
        chunk_overlap=chunkoverlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(pages)
    return docs


def qdrant_instance_func(db_client, embeddings, user_id):
    return qdrant.Qdrant(
        client=db_client,
        collection_name=user_id,
        embeddings=embeddings
    )


def create_embeddings(chunks, db_client, embeddings, user_id, sessionid):
    print("creating embeddings with Qdrant")
    qdrant_instance = qdrant_instance_func(db_client=db_client, embeddings=embeddings, user_id=user_id)
    text = [t.page_content for t in chunks]
    metadata = [{"source": m.metadata['source'], "sessionid": sessionid} for m in chunks]
    qdrant_instance.from_texts(texts=text, metadatas=metadata, embedding=embeddings, collection_name=user_id)


def similarity_search(db_client, embeddings, query, user_id):
    found_docs = qdrant_instance_func(db_client=db_client, embeddings=embeddings, user_id=user_id).similarity_search(
        query=query, k=2)
    print(found_docs)
    print(f" Total documents found {len(found_docs)} ,   because k is selected to 2")
    return found_docs


def doc_retriever(db_client, similar_docs, embeddings, user_id):
    ret = qdrant_instance_func(db_client=db_client, embeddings=embeddings, user_id=user_id).as_retriever(
        search_type="similarity",
        search_kwargs={'filter': {
            'source': similar_docs[
                0].metadata['source']}})
    return ret


def generating_response(llm, query, template, retriever):
    qa_prompt = PromptTemplate(input_variables=["query"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=qa_prompt,
        verbose=True,
    )
    response = qa.run(query)
    return response


def delete_embeddings_by_id(db_client, document_name, user_id):
    collection_name = user_id
    metadata_field = "metadata.source"
    metadata_value = document_name

    db_client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key=metadata_field,
                        match=MatchValue(value=metadata_value),
                    ),
                ],
            )
        ),
    )


def delete_collection(db_client, user_id):
    db_client.delete_collection(collection_name=user_id)
