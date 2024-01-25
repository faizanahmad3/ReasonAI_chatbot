from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Anyscale
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# embedding model
embedding_model_name = "BAAI/bge-large-en-v1.5"
embedding_model_kwargs = {'device': 'cpu'}
embedding_encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)

# LLM model

# ANYSCALE_MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.1'
# # ANYSCALE_MODEL_NAME = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
# anyscale_api_base = os.getenv("ANYSCALE_API_BASE")
# ayscale_api_key = os.getenv("ANYSCALE_API_KEY")
#
# mistral_llm = Anyscale(model_name=ANYSCALE_MODEL_NAME, anyscale_api_base=anyscale_api_base,
#                        ayscale_api_key=ayscale_api_key)

mistral_llm = ""
