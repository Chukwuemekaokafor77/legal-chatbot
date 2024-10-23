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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define file paths
EMBEDDINGS_PATH = "./pdf_embeddings.json"
TEXTS_PATH = "./pdf_texts.json"
FAISS_INDEX_PATH = "./legal_docs_index.faiss"
LEGALBERT_MODEL_PATH = "./legalbert_finetuned"  # Path to the fine-tuned LegalBERT
LEGAL_T5_SUMMARIZATION_PATH = "t5-small"  # For summarization
LEGAL_T5_CLASSIFICATION_PATH = "SEBIS/legal_t5_small_cls_en"  # For classification

# Valid years with data in the database
VALID_YEARS = list(range(2011, 2018)) + [2021, 2022]

def verify_files():
    files_to_check = [EMBEDDINGS_PATH, TEXTS_PATH, FAISS_INDEX_PATH, LEGALBERT_MODEL_PATH]
    for file in files_to_check:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}")

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

        # Debug prints
        print(f"Number of documents loaded: {len(self.texts_data)}")
        print(f"FAISS index size: {self.index.ntotal}")
    
    def _prepare_bm25(self):
        if isinstance(self.texts_data, dict):
            corpus = [str(doc).lower() for doc in self.texts_data.values()]
        elif isinstance(self.texts_data, list):
            corpus = [str(doc).lower() for doc in self.texts_data]
        else:
            logger.error(f"Unexpected texts_data type: {type(self.texts_data)}")
            raise ValueError(f"Unexpected texts_data type: {type(self.texts_data)}")
        
        tokenized_corpus = [doc.split() for doc in corpus]
        return BM25Okapi(tokenized_corpus)

# Define the RAGModel class (with year filtering)
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
        print(f"Top 5 BM25 scores: {sorted(bm25_scores, reverse=True)[:5]}")
        top_n = min(top_k * 2, len(bm25_scores))  # Get more candidates for filtering
        top_indices = np.argsort(bm25_scores)[-top_n:][::-1]

        results = []
        for idx in top_indices:
            doc_id = filtered_ids[idx]
            doc = self.document_store.texts_data[doc_id]

            # Check if the document is a dictionary or a string
            if isinstance(doc, dict):
                # Document is a dictionary, so safely use .get()
                doc_text = doc.get('Title', '') + '\n' + doc.get('Citation', '')
            elif isinstance(doc, str):
                # Document is a string, so use it directly as text
                doc_text = doc  # Use the document string as is
            else:
                # If the document type is unexpected, log an error and skip
                logger.error(f"Unexpected document type: {type(doc)} for doc_id: {doc_id}")
                continue

            # Append the result
            results.append({
                "id": doc_id,
                "text": doc_text,
                "similarity": bm25_scores[idx]
            })

            if len(results) == top_k:
                break

        print(f"Number of results after retrieval: {len(results)}")
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

# Define the LegalT5Classifier class
class LegalT5Classifier:
    def __init__(self, model_name: str = LEGAL_T5_CLASSIFICATION_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelWithLMHead.from_pretrained(model_name)

    def classify(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        output = self.model.generate(**inputs)
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
    def __init__(self, embeddings_path: str, texts_path: str, faiss_index_path: str):
        self.document_store = DocumentStore(embeddings_path, texts_path, faiss_index_path)
        self.rag_model = RAGModel(self.document_store)
        self.legalbert_response_generator = LegalBERTResponseGenerator()
        self.summarizer = LegalT5Summarizer()  # Integrate LegalT5 summarizer
        self.classifier = LegalT5Classifier()  # Integrate LegalT5 classifier

    def process_query(self, query: str, year: int = None) -> Dict[str, Any]:
        try:
            # Get relevant documents based on the query and optional year filter
            relevant_docs = self.rag_model.get_relevant_documents(query, year=year)
            if not relevant_docs:
                return {
                    "query": query,
                    "error": "No relevant documents found"
                }

            # Generate summary for each document using LegalT5
            for doc in relevant_docs:
                doc['summary'] = self.summarizer.summarize(doc['text'])

            # Classify each document using LegalT5 classification
            for doc in relevant_docs:
                doc['classification'] = self.classifier.classify(doc['text'])

            # Generate a response using the fine-tuned LegalBERT model
            response = self.legalbert_response_generator.generate_response(query, relevant_docs)

            return {
                "query": query,
                "relevant_documents": relevant_docs,
                "response": response
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "error": str(e)
            }

# Initialize RAG System
def initialize_rag_system():
    try:
        verify_files()
        rag_system = RAGSystem(EMBEDDINGS_PATH, TEXTS_PATH, FAISS_INDEX_PATH)
        logger.info("RAG System initialized successfully.")
        return rag_system
    except Exception as e:
        logger.error(f"Failed to initialize RAG System: {e}")
        st.error(f"Failed to initialize RAG System: {e}")
        return None

# Streamlit App with Chatbot Interface
def main():
    st.title("Canadian Law RAG System")
    st.write("Powered by Themiscore.")

    # Initialize RAG system
    rag_system = initialize_rag_system()
    if not rag_system:
        st.stop()

    # Chatbot Interface
    st.header("Chatbot Interface")
    user_input = st.text_input("Ask a legal question:")
    year_input = st.selectbox("Select Year (optional)", [""] + VALID_YEARS)

    if st.button("Submit"):
        if not user_input.strip():
            st.warning("Please enter a valid question.")
        else:
            year = int(year_input) if year_input else None
            with st.spinner("Processing your query..."):
                start_time = time.time()
                try:
                    result = rag_system.process_query(query=user_input, year=year)
                    end_time = time.time()

                    if "error" in result:
                        st.error(result["error"])
                    elif not result['relevant_documents']:
                        st.warning("No relevant documents found. Try adjusting your query.")
                    else:
                        st.success("Query processed successfully!")
                        st.markdown(f"**Query:** {result['query']}")

                        st.markdown("### Relevant Documents:")
                        for i, doc in enumerate(result['relevant_documents'], 1):
                            # Display summary first
                            st.markdown(f"**Document {i}:** `{doc['id']}`")
                            st.markdown(f"*Similarity Score:* {doc['similarity']:.4f}")
                            st.markdown(f"*Summary:* {doc['summary']}")
                            st.markdown(f"*Classification:* {doc['classification']}")

                            # Add expander for the full document
                            with st.expander(f"View Full Document {i}"):
                                st.markdown(f"*Content:* {doc['text']}")
                            st.markdown("---")

                        st.markdown("### Response:")
                        st.write(result['response'])

                        st.markdown(f"**Processing Time:** {end_time - start_time:.2f} seconds")
                except Exception as e:
                    st.error(f"An error occurred while processing your query: {str(e)}")
                    logger.exception("Error in query processing")

if __name__ == "__main__":
    main()
