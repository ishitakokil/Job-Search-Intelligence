import streamlit as st
from workflow import create_workflow
from typing_extensions import TypedDict, List, Literal

class RAGState(TypedDict):
    query: str
    refined_query: str
    retrieved_docs: List[str]
    formatted_context: str
    llm_answer: str
    decision: Literal["relevant", "not_relevant", "useful", "not_useful", "end"]
    retries: int

st.set_page_config(page_title="Multi-Agent RAG System", layout="wide")
st.title(" Multi-Agent AI Job Search Assistant")

use_lora = st.toggle("Use Fine-Tuned LoRA Model for Summarization", value=False)
query = st.text_input("Enter your job search query:", "What are the highest paying remote Data Science jobs?")

if st.button("Search Jobs"):
    with st.spinner("Processing your query..."):
        app = create_workflow(use_lora=use_lora)

        input_state = RAGState({
            "query": query,
            "refined_query": "",
            "retrieved_docs": [],
            "formatted_context": "",
            "llm_answer": "",
            "decision": "",
            "retries": 0
        })

        output_state = app.invoke(input_state)
        output_state = dict(output_state)

        st.markdown(output_state.get("llm_answer", "No response generated."), unsafe_allow_html=True)

        with st.expander("📄 View Retrieved Job Listings"):
            st.markdown(output_state.get("formatted_context", "No job listings retrieved."), unsafe_allow_html=True)





