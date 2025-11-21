#importing required packages
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import streamlit as st
#reusable code for query-response
load_dotenv()

	# Load and split documents
loader = DirectoryLoader(r"C:\Users\rupsh\MDTM28\chatbot", glob= "*.pdf", loader_cls= PyPDFLoader)
documents = loader.load()
splited_text = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 4)
docs = splited_text.split_documents(documents)
# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 
# Initialize Pinecone instance        
pc = Pinecone(api_key= os.getenv("API_KEY"))
# Create Pinecone Index
index_name = "bank-docs"
if index_name not in pc.list_indexes().names():
		pc.create_index(
			name=index_name,
			dimension=768,
			metric="cosine",
			spec=ServerlessSpec(
				cloud="aws",
				region="us-east-1"
			)  
		)
index = pc.Index(index_name)
vectorstore = PineconeVectorStore(docs, embedding = embeddings, index=index)

#Intitialise GeminiAI LLLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
prompt = ChatPromptTemplate.from_messages([
		("system", "You are a helpful AI assistant. Use the following context to answer the question:\n\n{context}"),
		("human", "{input}")
	])
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
# Create the retrieval chain connecting retriever and QA chain
rag_chain = (retriever, combine_docs_chain)
#streamlit deployment	
st.title('Assistive bankbot')
#function to generate LLM response
def generate_response(user_input):
    result = rag_chain.invoke({"input": user_input})
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm a Bankbot. How can I assist you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate  response 
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Checking bank Documents..."):
            response = generate_response(input) 
            result_text = response.get("answer")
            st.write(result_text) 
    message = {"role": "assistant", "content": result_text}
    st.session_state.messages.append(message)
