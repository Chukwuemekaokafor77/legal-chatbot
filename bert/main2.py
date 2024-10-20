import streamlit as st
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelWithLMHead, BertTokenizer, BertForSequenceClassification
import time
from rank_bm25 import BM25Okapi
import requests

# Dropbox URLs
dropbox_links = {
    'pdf_embeddings': 'https://www.dropbox.com/scl/fi/2c6nl31rskv4t40uj58xo/pdf_embeddings.json?rlkey=6ekhg7lwgt40mb9vnnalabgqp&st=06srbgfo&dl=1',
    'pdf_texts': 'https://www.dropbox.com/scl/fi/mo5h5hx5w9zh60i13lokp/pdf_texts.json?rlkey=5y1d3u34xubn5yx8k38vz9o1n&st=u5v3jp8t&dl=1',
    'legal_docs_index': 'https://www.dropbox.com/scl/fi/ni1uwvdkz9gjekq72t7mn/legal_docs_index.faiss?rlkey=w9fafgc51qtzgoox6w0m97dxo&st=rfqok1ju&dl=1',
    'weakly_labeled_data': 'https://www.dropbox.com/scl/fi/6tz78wpa73y173nwjryjb/weakly_labeled_data.json?rlkey=v6c1moduprmetc60732wtrn3e&st=2h0z97f0&dl=1'
}

def download_file(url, output):
    response = requests.get(url, stream=True)
    with open(output, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

# Paths where files will be saved
file_paths = {
    'pdf_embeddings': 'pdf_embeddings.json',
    'pdf_texts': 'pdf_texts.json',
    'legal_docs_index': 'legal_docs_index.faiss',
    'weakly_labeled_data': 'weakly_labeled_data.json'
}

# Check if files already exist, if not download them
for file_key, file_url in dropbox_links.items():
    file_path = file_paths[file_key]
    if not os.path.exists(file_path):
        print(f"Downloading {file_path} from Dropbox...")
        download_file(file_url, file_path)
        print(f"{file_path} downloaded successfully.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define file paths
EMBEDDINGS_PATH = file_paths['pdf_embeddings']
TEXTS_PATH = file_paths['pdf_texts']
FAISS_INDEX_PATH = file_paths['legal_docs_index']
LEGALBERT_MODEL_PATH = "./legalbert_finetuned"  # Path to the fine-tuned LegalBERT
LEGAL_T5_SUMMARIZATION_PATH = "t5-small"  # For summarization
LEGAL_T5_CLASSIFICATION_PATH = "SEBIS/legal_t5_small_cls_en"  # For classification

# Load pdf_embeddings.json
with open(EMBEDDINGS_PATH, 'r') as f:
    pdf_embeddings = json.load(f)

# Load pdf_texts.json
with open(TEXTS_PATH, 'r') as f:
    pdf_texts = json.load(f)

# Load legal_docs_index.faiss (FAISS index)
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# Load weakly_labeled_data.json
with open(file_paths['weakly_labeled_data'], 'r') as f:
    weakly_labeled_data = json.load(f)

# Valid years with data in the database
VALID_YEARS = list(range(2011, 2018)) + [2021, 2022]

# Define the DocumentStore class
class DocumentStore:
    def __init__(self, embeddings_path: str, texts_path: str, faiss_index_path: str):
        self.embeddings_path = embeddings_path
        self.texts_path = texts_path
        self.faiss_index_path = faiss_index_path
        
        # Load FAISS index
        if os.path.exists(faiss_index_path):
            self.index = faiss.read_index(faiss_index_path)
            logger.info("FAISS index loaded successfully.")
        else:
            logger.error(f"FAISS index file not found at {faiss_index_path}.")
            raise FileNotFoundError(f"FAISS index file not found at {faiss_index_path}.")
        
        # Initialize sentence transformer for query encoding
        self.model = SentenceTransformer('all-mpnet-base-v2')
        logger.info("SentenceTransformer model loaded.")

        # Load texts data
        if os.path.exists(texts_path):
            with open(self.texts_path, 'r', encoding='utf-8') as f:
                self.texts_data = json.load(f)
            logger.info("Texts data loaded successfully.")
        else:
            logger.error(f"Texts data file not found at {texts_path}.")
            raise FileNotFoundError(f"Texts data file not found at {texts_path}.")
        
        # Prepare BM25 index for full-text search
        self.bm25 = self._prepare_bm25()
        logger.info("BM25 index prepared.")
    
    def _prepare_bm25(self):
        corpus = [str(doc).lower() for doc in self.texts_data.values()]
        tokenized_corpus = [doc.split() for doc in corpus]
        return BM25Okapi(tokenized_corpus)

# Define the RAGModel class
class RAGModel:
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store

    def get_relevant_documents(self, query: str, top_k: int = 3, year: int = None) -> List[Dict[str, Any]]:
        print(f"Query: {query}")
        print(f"Year filter: {year}")
        
        # Filter by year (if provided)
        filtered_docs = []
        for doc_id, doc in self.document_store.texts_data.items():
            if isinstance(doc, dict) and 'Publication Date' in doc:
                doc_year = int(doc['Publication Date'].split('-')[0])
                if year is None or doc_year == year:
                    filtered_docs.append((doc_id, doc))
            elif isinstance(doc, str):
                filtered_docs.append((doc_id, doc))

        # Full-text search using BM25 on filtered documents
        filtered_ids = [doc[0] for doc in filtered_docs]
        filtered_corpus = [str(doc[1]) for doc in filtered_docs]
        tokenized_query = query.lower().split()
        bm25 = BM25Okapi([doc.lower().split() for doc in filtered_corpus])
        bm25_scores = bm25.get_scores(tokenized_query)
        top_n = np.argsort(bm25_scores)[-top_k:][::-1]

        results = []
        for idx in top_n:
            doc_id = filtered_ids[idx]
            doc = self.document_store.texts_data[doc_id]
            doc_text = doc.get('Title', '') + '\n' + doc.get('Citation', '') if isinstance(doc, dict) else doc
            results.append({
                "id": doc_id,
                "text": doc_text,
                "similarity": bm25_scores[idx]
            })

        return results

# Define the LegalT5Summarizer class
class LegalT5Summarizer:
    def __init__(self, model_name: str = LEGAL_T5_SUMMARIZATION_PATH):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def summarize(self, text: str, max_length: int = 100) -> str:
        input_ids = self.tokenizer.encode("summarize: " + text[:512], return_tensors="pt", max_length=512, truncation=True)
        output = self.model.generate(input_ids, max_length=max_length, num_beams=2, early_stopping=True)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# Define the LegalBERTResponseGenerator class
class LegalBERTResponseGenerator:
    def __init__(self, model_path: str = LEGALBERT_MODEL_PATH):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

    def generate_response(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        context = "\n\n".join([
            f"Document {i+1} ({doc['id']}, Similarity: {doc['similarity']:.4f}):\n{doc['text']}" 
            for i, doc in enumerate(relevant_docs)
        ])
        
        prompt = f"{context}\nQuery: {query}\nResponse:"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            output = self.model(**inputs)
            logits = output.logits
            probabilities = logits.softmax(dim=-1)
            predicted_class = probabilities.argmax().item()
            confidence = probabilities[0, predicted_class].item()
            
            class_names = ["Irrelevant", "Relevant"]
            response = f"The query is classified as {class_names[predicted_class]} with confidence {confidence:.2f}"
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

# Define the RAGSystem class
class RAGSystem:
    def __init__(self):
        self.document_store = DocumentStore(EMBEDDINGS_PATH, TEXTS_PATH, FAISS_INDEX_PATH)
        self.rag_model = RAGModel(self.document_store)
        self.legalbert_response_generator = LegalBERTResponseGenerator()
        self.summarizer = LegalT5Summarizer()

    def process_query(self, query: str, year: int = None) -> Dict[str, Any]:
        try:
            # Get relevant documents
            relevant_docs = self.rag_model.get_relevant_documents(query, year=year)
            if not relevant_docs:
                return {"query": query, "error": "No relevant documents found"}
            
            # Summarize each document
            for doc in relevant_docs:
                doc['summary'] = self.summarizer.summarize(doc['text'])
            
            # Generate a response using the fine-tuned LegalBERT
            response = self.legalbert_response_generator.generate_response(query, relevant_docs)

            return {
                "query": query,
                "relevant_documents": relevant_docs,
                "response": response
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"query": query, "error": str(e)}

# Initialize and run the Streamlit app
def main():
    st.title("Canadian Law RAG System")
    st.write("Interactive application to query Canadian jurisprudence and interact with the chatbot.")

    # Initialize RAG system
    rag_system = RAGSystem()

    # User input
    query = st.text_input("Ask a legal question:")
    year = st.selectbox("Select Year (optional)", [""] + VALID_YEARS)

    if st.button("Submit"):
        if not query.strip():
            st.warning("Please enter a valid question.")
        else:
            year = int(year) if year else None
            with st.spinner("Processing your query..."):
                result = rag_system.process_query(query=query, year=year)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Query processed successfully!")
                    st.markdown(f"**Query:** {result['query']}")

                    for i, doc in enumerate(result['relevant_documents'], 1):
                        st.markdown(f"**Document {i}:** `{doc['id']}`")
                        st.markdown(f"*Similarity Score:* {doc['similarity']:.4f}")
                        st.markdown(f"*Summary:* {doc['summary']}")

                        with st.expander(f"View Full Document {i}"):
                            st.write(doc['text'])

                    st.markdown("### Response:")
                    st.write(result["response"])

if __name__ == "__main__":
    main()
