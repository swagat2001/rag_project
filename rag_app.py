import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

def load_document(pdf_path):
    print(f"Loading documant from:{pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        separators = ["\n\n","\n","."," "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into{len(chunks)} chunks")
    return chunks

def create_vectorstore(chunks):
    print("Creating embeddings and storing in chroma")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    import time
    
    batch_size = 80
    all_chunks_done = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing chunks {i} to {i + len(batch)} of {len(chunks)}...")
        
        if i == 0:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
        else:
            vectorstore.add_documents(batch)
        
        if i + batch_size < len(chunks):
            print(f"Rate limit pause — waiting 65 seconds...")
            time.sleep(65)
    print("Vector store created successsfully")
    return vectorstore

def build_rag_chain(vectorstore):
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    from langchain_core.prompts import PromptTemplate
    
    prompt_template = """
    You are a helpful assistant for NISM certification exam preparation.
    Use only the following context to answer the question.
    If the answer is not found in the context, say 
    "I don't have enough information in the document to answer this."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    return retriever, prompt, llm

if __name__ == "__main__":
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    print(f"Loaded vector store. Total chunks: {vectorstore._collection.count()}")
    
    
    retriever, prompt, llm = build_rag_chain(vectorstore)
 
    query = "What is the role of a compliance officer?"
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    filled_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(filled_prompt)
    
    print(f"\nQuestion: {query}")
    print(f"\nAnswer: {response.content}")
    print(f"\nSources used:")
    for doc in retrieved_docs:
        print(f"  Page {doc.metadata['page']}: {doc.page_content[:150]}")