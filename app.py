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

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set page-wide styles
def set_styles():
    st.markdown(
        """
        <style>
        .block-container {
            padding: 20px;
            background-color: #00203FFF;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .block-header {
            padding: 10px;
            background-color: #4682B4;
            color: white;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }
        .block-content {
            margin-top: 20px;
            color: #1E0342 !important; /* Change text color to blue */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='color: #ADEFD1FF'>Blog Generation and PDF Chat Application</h1>", unsafe_allow_html=True)
    



# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to handle user input and provide response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

# Function to extract vocabulary from text chunks
def extract_vocabulary(text_chunks):
    words = ' '.join(text_chunks).split()
    return set(words)

# Function to extract term frequency from text chunks
def extract_term_frequency(text_chunks):
    words = ' '.join(text_chunks).split()
    term_frequency = Counter(words)
    return term_frequency

# Function to save vocabulary to a JSON file
def save_vocabulary(vocabulary, file_path):
    with open(file_path, 'w') as f:
        json.dump(list(vocabulary), f)

# Function to save term frequency to a JSON file
def save_term_frequency(term_frequency, file_path):
    with open(file_path, 'w') as f:
        json.dump(term_frequency, f)

# Function to save vocabulary and term frequency statistics
def save_statistics(text_chunks, vocabulary_file_path, term_frequency_file_path):
    vocabulary = extract_vocabulary(text_chunks)
    term_frequency = extract_term_frequency(text_chunks)
    save_vocabulary(vocabulary, vocabulary_file_path)
    save_term_frequency(term_frequency, term_frequency_file_path)

# Function to generate a blog based on input parameters
def generate_blog(input_text, no_words, blog_style):
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

# Function to get conversational chain for question answering
def get_conversational_chain():
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

# Main function
def main():
    set_styles()
    #st.title("Blog Generation and PDF Chat Application")

    option = st.sidebar.radio("Choose an option:", ("Generate Blog", "Interact with PDF via Chat"))

    if option == "Generate Blog":
        st.subheader("Generate Blog 🤖")

        input_text = st.text_input("Enter the Blog Topic")
        no_words = st.slider('No of Words', min_value=50, max_value=1000, value=250, step=50)
        blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'))

        submit = st.button("Generate Blog")

        if submit:
            st.subheader("Generated Blog:")
            st.markdown(f"### Topic: {input_text}")
            st.markdown(
                f'<div style="color: #ADEFD1FF">{generate_blog(input_text, no_words, blog_style)}</div>',
                unsafe_allow_html=True
            )

    elif option == "Interact with PDF via Chat":
        st.subheader("Chat with PDF💬")

        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
            if st.button("Process PDFs"):
                if pdf_docs:
                    with st.spinner("Processing PDFs..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs Processed Successfully")
                        save_statistics(text_chunks, 'vocabulary.json', 'term_frequency.json')

if __name__ == "__main__":
    main()
