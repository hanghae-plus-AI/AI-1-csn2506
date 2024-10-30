import bs4
import streamlit as st
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# .env 파일에서 API 키 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# print(openai_api_key)

if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")


# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

loader = WebBaseLoader(
    web_paths=("https://spartacodingclub.kr/blog/all-in-challenge_winner",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("editedContent", "css-j3idia")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()  
)

retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = hub.pull("rlm/rag-prompt")

# Streamlit 앱 구성
st.title("LangChain RAG 기반 챗봇")
user_input = st.text_input("질문을 입력하세요:")


if user_input:
    # RAG 체인 사용하여 응답 생성
    retrieved_docs = retriever.invoke(user_input)
    formatted_docs = format_docs(retrieved_docs)
    user_prompt = prompt.invoke({"context": formatted_docs, "question": user_input})
    response = llm.invoke(user_prompt)
    st.write("응답:", response.content)

    from langchain_core.output_parsers import StrOutputParser
    final_response = StrOutputParser().parse(response.content)
    st.write("최종 응답:", final_response)

