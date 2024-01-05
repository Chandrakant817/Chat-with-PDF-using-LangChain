import os
import openai
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore")

# Set up Azure OpenAI configuration
deployment_name =  "gpt-35-turbo"
os.environ["OPENAI_API_KEY"] = "2513f9b32234498690f592b325e7180a"
os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_BASE'] = "https://demoazureopenai01.openai.azure.com/"
os.environ['OPENAI_API_VERSION'] = "2023-08-01-preview"

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask from your PDF")
    
    # Upload file
    pdf = st.file_uploader("Upload the PDF", type="pdf")
    
    # Extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        st.write(text[:100])
        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=70,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.write(chunks)
        # Pass the text chunks to the Embedding Model from Azure OpenAI API to generate embeddings.
        embeddings = OpenAIEmbeddings(
            openai_api_key="",  # Replace with your API key
            deployment="text-embedding-ada-002", 
            client="azure"
        )

        # Create a vector store using FAISS
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        template = """
        Use the following pieces of context to answer the question at the end.
        If the question is not related to PDF, say that it is not related to PDF,
        strictly don't try to generate extra, irrelevant, or unwanted answers. Understand table values also. Always say "thanks for asking!" at the end of the answer.
        PDF Context: {context}
        Question: {question}
        
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Show user input
        user_question = st.text_input("Ask a question from your PDF:")
        
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = AzureOpenAI(
                temperature=0,
                openai_api_key="",  # Replace with your API key
                deployment_name="gpt-35-turbo",  # Adjust model name
                model_name="gpt-35-turbo"  # Adjust model name
            )

            chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_CHAIN_PROMPT)
            response = chain.run(input_documents=docs, question=user_question)
            answer = response  # You can use get_completion here if needed
            st.write(answer)
            st.write("Thanks for asking!")

if __name__ == '__main__':
    main()
