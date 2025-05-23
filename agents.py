from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from config import openai_api_key, pinecone_api_key
from pinecone import Pinecone
from typing_extensions import TypedDict, List, Literal
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load vector DB
pc = Pinecone(api_key=pinecone_api_key, environment="us-east-1")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index = pc.Index("job-postings")
vectorstore = PineconeVectorStore(index, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity")

# OpenAI LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0.7)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Define Model Path & Offload Location
MODEL_PATH = "./lora/lora_finetuned_model"
offload_dir = "./lora/offload_weights"  # Offload location for large models

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16, 
    device_map="auto",  # Auto-assign layers to GPU/CPU
    offload_folder=offload_dir  # Ensure offloaded layers go here
)

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


class RAGState(TypedDict):
    query: str
    refined_query: str
    retrieved_docs: List[str]
    formatted_context: str
    llm_answer: str
    decision: Literal["relevant", "not_relevant", "useful", "not_useful", "end"]
    retries: int

def query_rewriter(state: RAGState) -> RAGState:
    if state["retries"] >= 3:
        return {**state, "decision": "end"}
    refined_query = llm.predict(f"Rewrite the following query to improve search results:\n{state['query']}")
    return {**state, "refined_query": refined_query, "retries": state["retries"] + 1}

def retriever_agent(state: RAGState) -> RAGState:
    docs = retriever.invoke(state["refined_query"])
    if not docs:
        return {**state, "retrieved_docs": [], "formatted_context": "**No job listings found.**", "decision": "not_relevant"}
    job_listings = []
    for doc in docs:
        metadata = doc.metadata
        job_listings.append(
            f"**{metadata.get('title', 'N/A')}** at **{metadata.get('company_name', 'N/A')}**\n"
            f"Location: {metadata.get('location', 'N/A')}\n"
            f"Experience Level: {metadata.get('experience_level', 'N/A')}\n"
            f"Work Type: {metadata.get('work_type', 'N/A')}\n"
            f"Salary: {metadata.get('salary', 'N/A')}\n"
        )
    return {**state, "retrieved_docs": [doc.page_content for doc in docs], "formatted_context": "\n\n".join(job_listings)}

def grade_documents(state: RAGState) -> RAGState:
    if not state["formatted_context"] or state["formatted_context"] == "**No job listings found.**":
        return {**state, "decision": "not_relevant"}
    return {**state, "decision": "relevant"}

def generate_agent(state: RAGState) -> RAGState:
    prompt = (
        f"You are an AI job assistant. Below are job listings that match the query.\n"
        f"Summarize them and highlight the best options.\n\n"
        f"---\n\n{state['formatted_context']}\n\n---\n\nUser Query: {state['query']}\n"
    )
    answer = llm.predict(prompt)
    return {**state, "llm_answer": answer}

def generate_agent_lora(state: RAGState) -> RAGState:
    prompt = (
        "Here is a job description:\n\n"
        f"{state['formatted_context']}\n\n"
        "Provide a summary in one or two sentences:\n\n"
    )
    try:
        generated = text_generator(
            prompt,
            max_new_tokens=75,
            min_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        full_text = generated[0]['generated_text'].strip()
        summary = full_text.replace(prompt, "").strip().replace(state["formatted_context"], "").strip()
        return {**state, "llm_answer": summary}
    except Exception as e:
        return {**state, "llm_answer": f"Error generating summary: {e}"}

def verification_agent(state: RAGState) -> RAGState:
    check_prompt = (
        f"Compare the following AI-generated response to the retrieved job listings.\n"
        f"Does the response accurately reflect the retrieved job listings?\n"
        f"Respond with only 'useful' or 'not_useful'.\n\n"
        f"Job Listings:\n{state['formatted_context']}\n\n"
        f"AI Response:\n{state['llm_answer']}"
    )
    verification = llm.predict(check_prompt).strip().lower()
    return {**state, "decision": verification}

__all__ = [
    "query_rewriter", "retriever_agent", "grade_documents",
    "generate_agent", "generate_agent_lora", "verification_agent"
]

