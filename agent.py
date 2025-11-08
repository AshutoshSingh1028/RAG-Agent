import os
from typing import List, TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# ENVIRONMENT AND API KEY SETUP 

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""



# CORE COMPONENTS INITIALIZATION 

# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# Initialize Embeddings
embeddings = OpenAIEmbeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot-2" 

# DATA LOADING AND PREPARATION 

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of document objects, return a new list of document objects containing only 'source' in metadata
    and the original page_content"""
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    'source': src
                }
            )
        )
    return minimal_docs

def text_split(minimal_docs: List[Document]):
    """
    Split documents into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    return text_splitter.split_documents(minimal_docs)

def load_and_embed_data(pdf_path="Medical_book.pdf"):
    """
    One-time function to load, split, and embed the PDF into Pinecone.
    Checks if the index exists first.
    """
    print(f"Loading data from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    extracted_data = loader.load()
    print(f"Number of pages in document: {len(extracted_data)}")
    
    minimal_docs = filter_to_minimal_docs(extracted_data)
    text_chunk = text_split(minimal_docs)
    print(f"Split into {len(text_chunk)} chunks.")

    print(f"Creating or updating Pinecone index: {index_name}...")
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created. Embedding documents...")
        PineconeVectorStore.from_documents(
            documents=text_chunk,
            embedding=embeddings,
            index_name=index_name
        )
    else:
        print("Index already exists. Assuming data is populated.")
        
    print("Data loading complete.")

# LANGGRAPH AGENT DEFINITION 

# Graph State definition
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    documents: List[str]
    answer: str
    reflection: str
    plan_decision: str
    answer_attempts: int

# Plan node
def plan_node(state: GraphState):
    """
    Node 1: Plan
    Decides whether retrieval is needed based on the question.
    """
    print("--- 1. (PLAN) ---")
    question = state["question"]
    
    plan_prompt_template = """
    You are a planner for a RAG chatbot. Your knowledge base is about medical topics.
    Based on the user's question, decide if the bot needs to retrieve documents from
    its knowledge base or if it can answer directly.
    
    If the question is about medical topics (like disease, drugs, side effects, treatments),
    respond with the single word: 'retrieve'.
    
    If the question is a general greeting, a simple conversation, or off-topic,
    respond with the single word: 'direct_answer'.

    User Question: {question}
    Decision:
    """
    plan_prompt = PromptTemplate.from_template(plan_prompt_template)
    plan_chain = plan_prompt | llm | StrOutputParser()
    decision = plan_chain.invoke({"question": question})
    decision = decision.strip().lower().replace(".", "")
    
    print(f"Plan decision: {decision}")
    return {"plan_decision": decision, "documents": []} # Initialize documents

# Retrieve node
def retrieve_node(state: GraphState):
    """
    Node 2: Retrieve
    Performs RAG using the Pinecone vector database.
    """
    print("--- 2. (RETRIEVE) ---")
    question = state["question"]
    print(f"Retrieving documents for: {question}")
    
    try:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3}) # Get top 3
        docs = retriever.invoke(question)
        doc_contents = [d.page_content for d in docs]
        print(f"Found {len(doc_contents)} relevant documents.")
        return {"documents": doc_contents}

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return {"documents": []}

# Answer node
def answer_node(state: GraphState):
    """
    Node 3: Answer
    Generates an answer using the LLM.
    """
    print("--- 3. (ANSWER) ---")
    question = state["question"]
    documents = state["documents"]
    attempts = state.get("answer_attempts", 0)
    
    base_rag_prompt = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer from the context, just say that you don't know.
    Be concise.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    base_direct_prompt = "You are a helpful assistant. Answer the user's question. Question: {question}\nAnswer:"
    
    retry_prefix = ""
    if attempts > 0:
        retry_prefix = "The previous answer was not relevant. Please try again, focusing strictly on the user's question and the provided context (if any).\n\n"

    if documents:
        print(f"Generating answer using RAG (Attempt {attempts + 1})...")
        prompt_template = retry_prefix + base_rag_prompt
        prompt = PromptTemplate.from_template(prompt_template)
        context_string = "\n\n".join(documents)
        rag_chain = prompt | llm | StrOutputParser()
        answer = rag_chain.invoke({"context": context_string, "question": question})
    else:
        print(f"Generating direct answer (Attempt {attempts + 1})...")
        prompt_template = retry_prefix + base_direct_prompt
        prompt = PromptTemplate.from_template(prompt_template)
        direct_chain = prompt | llm | StrOutputParser()
        answer = direct_chain.invoke({"question": question})
    
    print(f"Generated answer: {answer}")
    return {"answer": answer, "answer_attempts": attempts + 1}

# Reflect node
def reflect_node(state: GraphState):
    """
    Node 4: Reflect
    Evaluates the answer for relevance.
    """
    print("--- 4. (REFLECT) ---")
    question = state["question"]
    answer = state["answer"]

    reflect_prompt_template = """
    You are a quality assurance assistant. Your job is to check if a
    generated answer is relevant to the user's question.

    Question:
    {question}

    Answer:
    {answer}

    Is the answer relevant to the question?
    Respond with only a single word: 'relevant' or 'not_relevant'.
    """
    reflect_prompt = PromptTemplate.from_template(reflect_prompt_template)
    reflect_chain = reflect_prompt | llm | StrOutputParser()
    
    print("Reflecting on answer quality...")
    reflection_result = reflect_chain.invoke({"question": question, "answer": answer})
    relevance = reflection_result.strip().lower().replace(".", "")
    
    print(f"Reflection result: {relevance}")
    return {"reflection": relevance}

# Decide node
def decide_next_node(state: GraphState):
    """
    Reads 'plan_decision' to decide the next node.
    """
    print("--- (ROUTING: Plan) ---")
    if state["plan_decision"] == "retrieve":
        print("Routing to: retrieve")
        return "retrieve"
    else:
        print("Routing to: generate_answer")
        return "generate_answer"

# Evaluate (Regenerate) node
MAX_ANSWER_ATTEMPTS = 2 # Set max attempts
def evaluate_answer(state: GraphState):
    """
    Reads 'reflection' and 'answer_attempts' to decide the next step.
    """
    print("--- (ROUTING: Reflect) ---")
    reflection = state["reflection"]
    attempts = state["answer_attempts"]
    
    if reflection == "relevant":
        print("Routing to: END (Answer is relevant)")
        return "end"
    else:
        print(f"Answer is not relevant (Attempt {attempts}).")
        if attempts >= MAX_ANSWER_ATTEMPTS:
            print(f"Routing to: END (Max attempts {MAX_ANSWER_ATTEMPTS} reached)")
            return "end"
        else:
            print("Routing to: generate_answer (Regenerating)")
            return "generate_answer"

# BUILD AND COMPILE THE GRAPH

print("Compiling agent graph...")
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("plan", plan_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate_answer", answer_node)
workflow.add_node("reflect", reflect_node)

# Define edges
workflow.set_entry_point("plan")
workflow.add_conditional_edges(
    "plan",
    decide_next_node,
    {"retrieve": "retrieve", "generate_answer": "generate_answer"}
)
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", "reflect")
workflow.add_conditional_edges(
    "reflect",
    evaluate_answer,
    {"generate_answer": "generate_answer", "end": END}
)

# Compile the graph
app = workflow.compile()
