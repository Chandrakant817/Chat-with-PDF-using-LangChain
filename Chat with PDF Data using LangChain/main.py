import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st

# Set up Azure OpenAI configuration
deployment_name = "gpt-35-turbo"
os.environ["OPENAI_API_KEY"] = "0e2b8bcd86b948b8b5fabe6f213cf221"
os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_BASE'] = "https://chandrakantopenai.openai.azure.com/"
os.environ['OPENAI_API_VERSION'] = "2023-09-15-previ"

def main():
    st.image("Screenshot 2023-11-10 160350.png", width=500)

    st.title("PDF Q/A App")

    st.sidebar.title("PDF Upload")
    st.sidebar.markdown("---")
    st.sidebar.image("pdf.png", width=75)

    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

    st.markdown("""
        <style>
            .footer {
                display: flex;
                justify-content: center;
                align-items: center;
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #FFA500;
                padding: 10px;
                margin-left: 120px;
            }
            .icon {
                display: inline-block;
                margin: 0 10px;
            }
        </style>
        <div class="footer">
            <!-- Your footer icons here -->
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
            .main {
                background-color: #FFFFE0;
                padding: 20px;
            }
        </style>
        <div class='main'>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        pdfReader = PdfReader(uploaded_file)
        raw_text = ''
        for i, page in enumerate(pdfReader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
    
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=70,
            length_function=len,
        )
        pdfTexts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings(openai_api_key="0e2b8bcd86b948b8b5fabe6f213cf221",
                                      deployment="text-embedding-ada-002",
                                      client="azure")

        knowledge_base = FAISS.from_texts(pdfTexts, embeddings)

        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. 
        Keep the answer as concise as possible. 
        Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        question = st.text_input("**Ask a question:**")
        if question:
            llm = AzureOpenAI(temperature=0,
                              openai_api_key="0e2b8bcd86b948b8b5fabe6f213cf221",
                              deployment_name="gpt-35-turbo",
                              model_name="gpt-35-turbo")

            docs = knowledge_base.similarity_search(question, k=1)

            qa_chain = RetrievalQA.from_chain_type(llm,
                                                   retriever=knowledge_base.as_retriever(),
                                                   return_source_documents=True,
                                                   chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

            result = qa_chain({"query": question})
            answer = result["result"][:-10]

            st.write(answer)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
