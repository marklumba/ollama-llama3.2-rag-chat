import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import SentenceTransformer, util
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import tempfile
import os
import pandas as pd

# UI Templates
bot_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px;">
    <div style="flex-shrink: 0; margin-right: 10px;">
        <img src="https://uxwing.com/wp-content/themes/uxwing/download/communication-chat-call/answer-icon.png" 
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div style="background-color: #e0e0e0; color: #333; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
</div>
'''

user_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: flex-end;">
    <div style="background-color: #007bff; color: white; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
    <div style="flex-shrink: 0; margin-left: 10px;">
        <img src="https://cdn.iconscout.com/icon/free/png-512/free-q-characters-character-alphabet-letter-36051.png?f=webp&w=512" 
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
</div>
'''

button_style = """
<style>
    .small-button {
        display: inline-block;
        padding: 5px 10px;
        font-size: 12px;
        color: white;
        background-color: #007bff;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-right: 5px;
    }
    .small-button:hover {
        background-color: #0056b3;
    }
</style>
"""


# File Handling Functions
def get_file_extension(file_name):
    """Get the file extension from the filename."""
    return os.path.splitext(file_name)[1].lower()

def create_temp_file(uploaded_file):
    """Create a temporary file from the uploaded file."""
    suffix = get_file_extension(uploaded_file.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

def load_document(file_path, file_extension):
    """Load document based on file extension."""
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.doc':
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        return loader.load()
    except Exception as e:
        st.error(f"Error loading file {file_path}: {str(e)}")
        return []

def prepare_and_split_docs(uploaded_files):
    """Prepare and split documents from multiple file formats."""
    split_docs = []
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
        disallowed_special=(),
        separators=["\n\n", "\n", " "]
    )
    
    for uploaded_file in uploaded_files:
        try:
            file_extension = get_file_extension(uploaded_file.name)
            temp_file_path = create_temp_file(uploaded_file)
            
            if file_extension == '.csv':
                df = pd.read_csv(temp_file_path)
                text_content = df.to_string(index=False)
                documents = [Document(page_content=text_content, metadata={"source": uploaded_file.name})]
            else:
                documents = load_document(temp_file_path, file_extension)
            
            if documents:
                split_docs.extend(splitter.split_documents(documents))
            
            os.unlink(temp_file_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    return split_docs

def create_file_uploader():
    """Create a file uploader that supports multiple file formats."""
    supported_formats = ['pdf', 'txt', 'csv', 'xlsx', 'xls', 'docx', 'doc']
    
    st.sidebar.markdown("### Upload Documents")
    st.sidebar.markdown("Supported formats: " + ", ".join(supported_formats))
    
    uploaded_files = st.sidebar.file_uploader(
        "Choose files",
        type=supported_formats,
        accept_multiple_files=True
    )
    
    if uploaded_files:
        total_files = len(uploaded_files)
        st.sidebar.markdown(f"📁 {total_files} file(s) uploaded")
        
        with st.sidebar.expander("View uploaded files"):
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({get_file_extension(file.name)})")
    
    return uploaded_files

# Vector Database Functions
def ingest_into_vectordb(split_docs):
    """Create and save vector database from documents."""
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(split_docs, embeddings)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db

# Conversation Chain Functions
def get_conversation_chain(retriever):
    """Create the conversation chain with the document retriever."""
    llm = OllamaLLM(model="llama3.2")
    
    # Context-aware question processing
    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "provide a response that directly addresses the user's query based on the provided documents. "
        "Do not rephrase the question or ask follow-up questions."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer generation
    system_prompt = (
        "As a personal chat assistant, provide accurate and relevant information based on the provided document in 2-5 sentences. "
        "Answer should be limited to 100 words and 2-5 sentences. Do not prompt to select answers or formulate stand-alone questions. "
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Chat history management
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

def calculate_similarity_score(answer: str, context_docs: list) -> float:
    """Calculate similarity between answer and context documents."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    context_docs = [doc.page_content for doc in context_docs]
    
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    context_embeddings = model.encode(context_docs, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(answer_embedding, context_embeddings)
    return similarities.max().item()

# Main Application
def main():
    st.title("📚 Multi-Format Document Chat")
    st.markdown(button_style, unsafe_allow_html=True)

    # Initialize session states
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'show_docs' not in st.session_state:
        st.session_state.show_docs = {}
    if 'similarity_scores' not in st.session_state:
        st.session_state.similarity_scores = {}

    # File upload and processing
    uploaded_files = create_file_uploader()

    if uploaded_files:
        if st.sidebar.button("Process Documents"):
            with st.spinner("Processing documents..."):
                split_docs = prepare_and_split_docs(uploaded_files)
                vector_db = ingest_into_vectordb(split_docs)
                retriever = vector_db.as_retriever()
                st.session_state.conversational_chain = get_conversation_chain(retriever)
                st.sidebar.success("✅ Documents processed successfully!")

    # Chat interface
    user_input = st.text_input("Ask a question about your documents:", placeholder="Type your question here...")

    if st.button("Submit"):
        if user_input and 'conversational_chain' in st.session_state:
            with st.spinner("Generating response..."):
                session_id = "abc123"  # You can make this dynamic
                response = st.session_state.conversational_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                context_docs = response.get('context', [])
                st.session_state.chat_history.append({
                    "user": user_input,
                    "bot": response['answer'],
                    "context_docs": context_docs
                })

    # Display chat history
    for index, message in enumerate(st.session_state.chat_history):
        # User message
        st.markdown(user_template.format(msg=message['user']), unsafe_allow_html=True)
        
        # Bot message
        st.markdown(bot_template.format(msg=message['bot']), unsafe_allow_html=True)

        # Initialize states for this message
        if f"show_docs_{index}" not in st.session_state:
            st.session_state[f"show_docs_{index}"] = False
        if f"similarity_score_{index}" not in st.session_state:
            st.session_state[f"similarity_score_{index}"] = None

        # Control buttons
        cols = st.columns([1, 1])
        
        with cols[0]:
            if st.button(f"📄 Show/Hide Sources", key=f"toggle_{index}"):
                st.session_state[f"show_docs_{index}"] = not st.session_state[f"show_docs_{index}"]

        with cols[1]:
            if st.button(f"🎯 Calculate Relevancy", key=f"relevancy_{index}"):
                if st.session_state[f"similarity_score_{index}"] is None:
                    score = calculate_similarity_score(message['bot'], message['context_docs'])
                    st.session_state[f"similarity_score_{index}"] = score

        # Show source documents if enabled
        if st.session_state[f"show_docs_{index}"]:
            with st.expander("Source Documents"):
                for doc in message.get('context_docs', []):
                    st.markdown(f"**Source:** {doc.metadata['source']}")
                    st.markdown(doc.page_content)

        # Display similarity score if calculated
        if st.session_state[f"similarity_score_{index}"] is not None:
            score = st.session_state[f"similarity_score_{index}"]
            score_color = "green" if score > 0.7 else "orange" if score > 0.5 else "red"
            st.markdown(f"Relevancy Score: <span style='color:{score_color}'>{score:.2f}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()