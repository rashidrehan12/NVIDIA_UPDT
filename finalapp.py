# import streamlit as st
# import os
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# import time

# from dotenv import load_dotenv 
# load_dotenv()

# ## load the Groq API key
# os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

# def vector_embedding():

#     if "vectors" not in st.session_state:

#         st.session_state.embeddings=NVIDIAEmbeddings()
#         st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) #splitting
#         print("hEllo")
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


    

# st.title("Nvidia NIM Demo")
# llm = ChatNVIDIA(model="meta/llama3-70b-instruct")


# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )


# prompt1=st.text_input("Enter Your Question From Doduments")


# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# import time



# if prompt1:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)
#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':prompt1})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")


import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv

# Set up the Streamlit app configuration
st.set_page_config(page_title="Nvidia NIM Demo", page_icon="🌅", layout="wide")

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Function to create vector embeddings
def vector_embedding():
    with st.spinner("Loading and processing documents..."):
        if "vectors" not in st.session_state:
            st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings
            st.success("Vector Store DB is ready!")

# # Custom CSS for styling
# st.markdown("""
#     <style>
#         /* Background gradient */
#         .main {
#             background: linear-gradient(to right, #ffecd2, #fcb69f);
#             color: #333333;
#         }
#         /* Header style */
#         .css-1v0mbdj {
#             background-color: #ff6f61;
#             color: white;
#             border-radius: 5px;
#         }
#         /* Sidebar style */
#         .css-18e3th9 {
#             background-color: #4a4e69;
#             color: white;
#             border-radius: 5px;
#         }
#         /* Textbox and button styles */
#         .stTextInput, .stButton button {
#             background-color: #6a0572;
#             color: white;
#             border-radius: 10px;
#             border: none;
#         }
#         /* Placeholder color */
#         ::placeholder {
#             color: #eeeeee;
#             opacity: 0.7;
#         }
#         /* Expander style */
#         .st-expander {
#             background-color: #9a031e;
#             color: white;
#             border-radius: 5px;
#         }
#         /* Footer style */
#         .footer {
#             position: fixed;
#             left: 0;
#             bottom: 0;
#             width: 100%;
#             background-color: #4a4e69;
#             color: white;
#             text-align: center;
#             padding: 10px;
#         }
#     </style>

# """, unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
    <style>
        /* Background gradient */
        .main {
            background: linear-gradient(to right, #ff9a9e, #fad0c4);
            color: #2c3e50;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Header style */
        .css-1v0mbdj {
            background-color: #ff6b81;
            color: white;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        /* Sidebar style */
        .css-18e3th9 {
            background-color: #34495e;
            color: #ecf0f1;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        /* Sidebar link hover effect */
        .css-18e3th9 a:hover {
            color: #f39c12;
            transform: scale(1.05);
            transition: all 0.3s ease-in-out;
        }
        /* Textbox and button styles */
        .stTextInput, .stButton button {
            background-color: #8e44ad;
            color: white;
            border-radius: 20px;
            border: none;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease;
        }
        .stTextInput:hover, .stButton button:hover {
            background-color: #9b59b6;
        }
        /* Placeholder color */
        ::placeholder {
            color: #bdc3c7;
            opacity: 0.8;
        }
        /* Expander style */
        .st-expander {
            background-color: #2ecc71;
            color: white;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            padding: 10px;
        }
        .st-expander-header {
            background-color: #27ae60;
            color: white;
            border-radius: 5px;
            padding: 5px;
            transition: all 0.3s ease-in-out;
        }
        .st-expander-header:hover {
            background-color: #2ecc71;
            transform: scale(1.02);
        }
        /* Footer style */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #34495e;
            color: #ecf0f1;
            text-align: center;
            padding: 15px;
            box-shadow: 0px -4px 10px rgba(0, 0, 0, 0.1);
        }
        .footer a {
            color: #f39c12;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        .footer a:hover {
            color: #f1c40f;
        }
    </style>
""", unsafe_allow_html=True)


# Title and description
st.title("Nvidia NIM Demo")
st.write("An interactive tool for embedding and querying documents using Nvidia's LLM.")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown("Use this sidebar to navigate through the app.")

# Nvidia Model Selection
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

# User input
st.subheader("Query the Documents")
prompt1 = st.text_input("Enter Your Question", placeholder="Ask a question based on the documents...")

# Button to process documents and create embeddings
if st.button("Create Document Embeddings"):
    vector_embedding()

# Display the query response
if prompt1:
    with st.spinner("Retrieving relevant information..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start_time
        
        st.write(f"Response time: {response_time:.2f} seconds")
        st.markdown("### Answer")
        st.success(response['answer'])

        # Displaying the context
        with st.expander("Document Similarity Search"):
            st.write("Relevant document excerpts:")
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Excerpt {i+1}:**")
                st.write(doc.page_content)
                st.write("---")

# Footer
st.markdown("""
    <div class="footer">
        Developed by Rashid Rehan
    </div>
""", unsafe_allow_html=True)
