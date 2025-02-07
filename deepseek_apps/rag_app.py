# importing necessary libraries
import streamlit as st 
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

# RAG Prompt 
PROMPT_TEMPLATE = """ 
You are an expert research assistant. Use the provided context to answer the user's questions.
If unsure, state that you don't kno. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {context}
Answer:
"""

PDF_PATH = 'document_store/pdfs/'

EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")

VECTOR_STORE = InMemoryVectorStore(embedding=EMBEDDING_MODEL)

LLM_MODEL = OllamaLLM(model="deepseek-r1:1.5b")


def save_uploaded_file(uploaded_file): 
    file_path = PDF_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_document(file_path):
    pdf_loader = PDFPlumberLoader(file_path)
    return pdf_loader.load()

def chunk_document(raw_document): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200, 
        add_start_index = True
    )
    return text_splitter.split_documents(raw_document)

def index_document(document_chunks): 
    docs_ids = VECTOR_STORE.add_documents(document_chunks)
    return docs_ids

def similarity_search(query, top_k=5): 
    return VECTOR_STORE.similarity_search(query, top_k)

def generate_answer(user_query, context): 
    context_text = "\n\n".join([doc.page_content for doc in context])
    rag_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    rag_chain = rag_prompt | LLM_MODEL
    return rag_chain.invoke({"user_query": user_query, "context": context_text})


# UI Configuration

st.title("Research Assistant Chatbot")
st.markdown("Upload a PDF document to get started.")
st.markdown("The chatbot will use the document to answer your questions.")
st.markdown("---")

# File Uploader
uploaded_pdf = st.file_uploader(
    "Upload a PDF document", 
    type="pdf", 
    help="Upload a PDF document to get started.",
    accept_multiple_files=False
)


if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_doc = load_pdf_document(saved_path)
    processed_chunks = chunk_document(raw_doc)
    indexed_ids = index_document(processed_chunks)
    print("Document Indexed Successfully, Document IDs: ", indexed_ids)

    st.success("Document Processed Successfully! You can now ask your questions.")

    user_input = st.chat_input("Enter your question:")

    if user_input:
        with st.chat_message("user"): 
            st.write(user_input)

        with st.spinner("Searching for answers..."):
            relevant_docs = similarity_search(user_input)
            llm_answer = generate_answer(user_input, relevant_docs)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(llm_answer)