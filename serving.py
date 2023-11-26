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

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)


# 사용할 LLM으로 openai의 gpt-3.5-turbo 활용
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0)

embedding=HuggingFaceEmbeddings(
model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",)
# model_kwargs = {'device': 'cuda'})
faiss_loaded = FAISS.load_local("index_union", embedding)
index = VectorStoreIndexWrapper(vectorstore = faiss_loaded)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/process_question', methods=['POST'])
def process_question():
    print("---------------retrievalQA 사용-------------------")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3, "fetch_k": 3}), verbose=True, return_source_documents=True)
    data = request.get_json()

    # 전달된 JSON 데이터에서 'question' 키의 값을 가져옴
    query = data['question']
    print(query)
    result = qa(query)

    print(result['result'])

    union_list = []

    for content in result['source_documents']:
        match = re.search(r'union_name:\s*([^\n]+)', content.page_content)
        if match:
            union_name = match.group(1).strip()
            union_list.append(union_name)

    print(union_list)

    response_data = {
        'answer': result['result'],
        'union_list': union_list  # union_list를 JSON 응답에 추가
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
