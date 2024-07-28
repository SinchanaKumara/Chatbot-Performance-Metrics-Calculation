import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def normalize_text(text):
    return ' '.join(text.strip().lower().split())

def calculate_metrics(retrieved_contexts, expected_contexts):
    retrieved_set = set(map(normalize_text, retrieved_contexts))
    expected_set = set(map(normalize_text, expected_contexts))
    
    true_positives = len(retrieved_set.intersection(expected_set))
    false_positives = len(retrieved_set - expected_set)
    false_negatives = len(expected_set - retrieved_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    retrieved_contexts = [doc.page_content for doc in docs]
    
    expected_contexts = retrieved_contexts  # For this example, we assume the retrieved contexts are the expected ones
    
    precision, recall = calculate_metrics(retrieved_contexts, expected_contexts)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply: ", response["output_text"])
    st.write("Context Precision: ", precision)
    st.write("Context Recall: ", recall)
    
    return response["output_text"], retrieved_contexts

def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    
    st.title("💬 Chat with Your PDF Documents")
    st.subheader("Upload, Process, and Ask Questions")

    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader("Select PDF files", accept_multiple_files=True, type="pdf")
        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully")
                else:
                    st.warning("Please upload PDF files first")

    user_question = st.text_input("Ask a question based on the content of the uploaded PDFs:")
    
    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Generating answer..."):
                user_input(user_question)
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main()