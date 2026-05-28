import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

AZURE_OPENAI_ENDPOINT = st.secrets[
    "AZURE_OPENAI_ENDPOINT"
]

AZURE_OPENAI_API_KEY = st.secrets[
    "AZURE_OPENAI_API_KEY"
]

AZURE_OPENAI_DEPLOYMENT = st.secrets[
    "AZURE_OPENAI_DEPLOYMENT"
]

AZURE_OPENAI_API_VERSION = st.secrets[
    "AZURE_OPENAI_API_VERSION"
]

AZURE_EMBEDDING_DEPLOYMENT = st.secrets[
    "AZURE_EMBEDDING_DEPLOYMENT"
]


# ==========================================
# Streamlit Page Config
# ==========================================

st.set_page_config(
    page_title="Diabetes AI Chatbot",
    page_icon="🩺",
    layout="centered"
)


# ==========================================
# Custom UI
# ==========================================

st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }

        .stTextInput > div > div > input {
            border-radius: 12px;
        }

        .stButton button {
            border-radius: 12px;
        }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# Title
# ==========================================

st.title("🩺 Diabetes AI Chatbot")
st.markdown("Ask any diabetes-related question from the knowledge base.")


# ==========================================
# Initialize Azure OpenAI LLM
# ==========================================

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0
)


# ==========================================
# Initialize HuggingFace Embeddings
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# ==========================================
# CSV File Path
# ==========================================

DATA_PATH = "/Users/prem/Desktop/AI_ML_Project/data/diabetes.csv"


# ==========================================
# Load Vector Store
# ==========================================

@st.cache_resource
def load_vectorstore():

    # Load CSV
    loader = CSVLoader(
        file_path=DATA_PATH,
        encoding="utf-8"
    )

    documents = loader.load()

    # Split Documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    # Create FAISS Vector Store
    vectorstore = FAISS.from_documents(
        docs,
        embedding_model
    )

    return vectorstore


# ==========================================
# Load Knowledge Base
# ==========================================

with st.spinner("Loading knowledge base..."):
    vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ==========================================
# User Input
# ==========================================

query = st.text_input(
    "Enter your question"
)


# ==========================================
# Generate Answer
# ==========================================

if query:

    with st.spinner("Generating answer..."):

        # Retrieve relevant chunks
        relevant_docs = retriever.invoke(query)

        # Combine context
        context = "\n\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        # Prompt
        prompt = f"""
        You are a diabetes specialist AI assistant.

        Answer the user's question ONLY from the provided context.

        If the answer is not present in the context,
        say:
        "Answer not found in the knowledge base."

        Context:
        {context}

        Question:
        {query}
        """

        # Generate response
        response = llm.invoke(prompt)

        answer = response.content

    # ==========================================
    # Display Answer
    # ==========================================

    st.markdown("### 💡 Answer")
    st.write(answer)
