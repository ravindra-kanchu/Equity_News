import os
import streamlit as st
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

st.title("Equity News Analysis📈")
st.sidebar.title("News Articles URLs")
# Generate embeddings using a local transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = 'faiss_store.pkl'
main_placeholder = st.empty()

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started...✅✅✅")
    data = loader.load()

    # Split text into chunks
    text_split = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '-', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter... Started...✅✅✅")
    docs = text_split.split_documents(data)
    
    
    embeddings = model.encode([doc.page_content for doc in docs])
    
    # Create FAISS vector store
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)
    
    # Save FAISS index and docs
    with open(file_path, 'wb') as f:
        pickle.dump((faiss_index, docs), f)

    # Display question input again after processing is done
    main_placeholder.text("Data loaded and indexed successfully! Now, ask your question.")

qa_pipeline = pipeline(
    "question-answering", 
    model="distilbert-base-cased-distilled-squad", 
    tokenizer="distilbert-base-cased-distilled-squad"
)

query = st.text_input("Question:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            faiss_index, docs = pickle.load(f)
        
        # Encode query
        query_embedding = model.encode([query])
        
        k = 3  # number of nearest neighbors to retrieve
        distances, indices = faiss_index.search(query_embedding, k)
        
        st.header("Answer")
        
        # Collect relevant content for QA
        relevant_content = " ".join([docs[idx].page_content for idx in indices[0]])
        
        # Pass content to QA model with increased context (up to max_length)
        qa_input = {
            "question": query,
            "context": relevant_content
        }
        
        # To get a more detailed answer, you could generate the response
        answer = qa_pipeline(qa_input)
        
        # Adjust output if needed
        st.write(answer['answer'])
        
        # Display sources without duplicates
        st.subheader("Sources:")
        unique_sources = set()
        for idx in indices[0]:
            source = docs[idx].metadata.get("source", "Unknown source")
            if source not in unique_sources:
                st.write(source)
                unique_sources.add(source)
