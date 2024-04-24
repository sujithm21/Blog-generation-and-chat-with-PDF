import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.llms import CTransformers
from collections import Counter
import json

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generate vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def user_input(user_question):
    """Handle user input and provide response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def extract_vocabulary(text_chunks):
    """Extract vocabulary from text chunks."""
    words = ' '.join(text_chunks).split()
    return set(words)

def extract_term_frequency(text_chunks):
    """Extract term frequency from text chunks."""
    words = ' '.join(text_chunks).split()
    term_frequency = Counter(words)
    return term_frequency

def save_vocabulary(vocabulary, file_path):
    """Save vocabulary to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(list(vocabulary), f)

def save_term_frequency(term_frequency, file_path):
    """Save term frequency to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(term_frequency, f)

def save_statistics(text_chunks, vocabulary_file_path, term_frequency_file_path):
    """Save vocabulary and term frequency statistics."""
    vocabulary = extract_vocabulary(text_chunks)
    term_frequency = extract_term_frequency(text_chunks)
    save_vocabulary(vocabulary, vocabulary_file_path)
    save_term_frequency(term_frequency, term_frequency_file_path)

def generate_blog(input_text, no_words, blog_style):
    """Generate a blog based on input parameters."""
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})

    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)

    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

def get_conversational_chain():
    """Get conversational chain for question answering."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def main():
    """Main function."""
    st.set_page_config("Chat PDF")
    st.header("Welcome to Blog Generation and PDF Chat Application")

    option = st.radio("Choose an option:", ("Generate Blog", "Interact with PDF via Chat"))

    if option == "Generate Blog":
        st.subheader("Generate Blog 🤖")

        input_text = st.text_input("Enter the Blog Topic")

        col1, col2 = st.columns([5, 5])
        with col1:
            no_words = st.text_input('No of Words')
        with col2:
            blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

        submit = st.button("Generate Blog")

        if submit:
            st.write(generate_blog(input_text, no_words, blog_style))

    elif option == "Interact with PDF via Chat":
        st.subheader("Chat with PDF using Gemini💁")

        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                    save_statistics(text_chunks, 'vocabulary.json', 'term_frequency.json')

if __name__ == "__main__":
    main()
