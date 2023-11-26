# 사용했던 package version

# !pip install openai==0.28.1
# !pip install langchain==0.0.316
# !pip install typing_extensions==4.6.3
# !pip install pydantic==1.10.9
# !pip install tiktoken
# !pip install faiss-cpu
# !pip install -U sentence-transformers
# !pip install python-dotenv

import os
import dotenv

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
import sentence_transformers
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import pandas as pd
import re


# union_data.csv 파일에서 동아리 데이터를 읽어옴
loader = CSVLoader("union_data.csv", encoding="utf-8")
data = loader.load()

# 사용할 LLM으로 openai의 gpt-3.5-turbo 활용
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)

# 앞서 읽어온 동아리 데이터를 KR-SBERT라는 임베딩 모델을 이용해 임베딩하고, FAISS 벡터스토어에 저장하여 index 생성
index = VectorstoreIndexCreator(
    vectorstore_cls=FAISS,
    embedding=HuggingFaceEmbeddings(
    model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    # model_kwargs = {'device': 'cuda'}
    ),
).from_loaders([loader])

# 만든 모델을 로컬에 저장하는 코드
# index.vectorstore.save_local("index_union")


print("---------------index.query 사용-------------------")
query = "농구 동아리 추천해줘"
result = index.query(query, llm = llm, verbose = True)
print(result)
print("--------------------------------------------------")


# 현재로서는 이 방법을 사용해야 할 것으로 판단됨.
print("---------------retrievalQA 사용-------------------")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3, "fetch_k": 3}), verbose = True, return_source_documents=True)

result = qa(query)

print(result['result'])

union_list = []

for content in result['source_documents']:
  match = re.search(r'union_name:\s*([^\n]+)', content.page_content)
  if match:
    union_name = match.group(1).strip()
    union_list.append(union_name)

print(union_list)
print("--------------------------------------------------")


# 아래와 같은 방법으로도 위에서 어떤 동아리가 추천되었는지를 알 수 있음.
print("------------------index.vectorstore.similarity_search 사용-------------------")
doc = index.vectorstore.similarity_search(query, k = 3, fetch_k = 3)
for d in doc:
  print(d.page_content)
  print()
print("-----------------------------------------------------------------------------")



# 저장한 모델을 불러와서 사용하는 방법

# embedding=HuggingFaceEmbeddings(
# model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
# model_kwargs = {'device': 'cuda'})
# faiss_loaded = FAISS.load_local("index_union", embedding)
# index_loaded = VectorStoreIndexWrapper(vectorstore = faiss_loaded)

# index_loaded.query(query, llm = llm, verbose = True)

# qa_loaded = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=faiss_loaded.as_retriever(search_kwargs={"k": 6, "fetch_k": 6}), verbose = True)

# qa_loaded.run(query)

# doc_loaded = index_loaded.vectorstore.similarity_search(query, k = 6, fetch_k = 6)
# print(doc_loaded[0])