
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="Assistant Règlement Drift", layout="wide")
st.title("📘 Assistant Règlement Drift 2025")

query = st.text_input("Posez votre question réglementaire ici :")

if query:
    with st.spinner("Recherche dans les règlements..."):
        loaders = [
            PyPDFLoader("Reglement-standard-Drift-2025.pdf"),
            PyPDFLoader("Reglement-Technique-Drift-2025.pdf"),
            PyPDFLoader("equipement-securite.pdf"),
            PyPDFLoader("Annexe-J-FIA 2024.pdf"),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(split_docs, embedding=embeddings)

        retriever = vectordb.as_retriever()
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

        result = qa_chain.run(query)
        st.markdown("### Réponse :")
        st.write(result)
