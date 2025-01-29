
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS         
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit title
st.title("Data Science Q&A")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Chat Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to create vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Input field for user query
prompt1 = st.text_input("What do you want to ask from the documents?")

# Button to create vector store
if st.button("Create Vector Store"):
    vector_embedding()
    st.write("Vector Store DB is ready!")

# Handle user queries
if prompt1:
    if "vectors" not in st.session_state:
        st.error("Please create the vector store first by clicking the button.")
    else:
        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Query the chain
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt1})
        end = time.process_time()

        # Display the response
        st.write(f"Answer: {response['answer']}")
        st.write(f"Response Time: {end - start:.2f} seconds")

        # Display document similarity search results
        if "context" in response:
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("----------------------------")
