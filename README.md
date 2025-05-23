# Advanced RAG (Retrieval Augmented Generation) System for Job Search Intelligence
This project showcases a multi-agent, Retrieval-Augmented Generation (RAG) pipeline designed to answer job search-related questions using a combination of document retrieval, fine-tuned summarization, and LLM-based reasoning. The system is evaluated across multiple configurations, from a base LLM to an advanced agentic RAG with a fine-tuned LoRA model.

## About the Data
The LinkedIn Job Postings dataset is a comprehensive collection of job listings sourced from LinkedIn. This dataset contains information about various job postings, including job titles, companies, locations, employment types, industries, and more. Link to dataset: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

## Toggleable LoRA Integration
The system is designed to allow dynamic workflow customization. Users can toggle between basic RAG and LoRA-enhanced RAG in both the backend workflow and the frontend UI. This flexibility enables comparisons between retrieval-only responses and LoRA fine-tuned summaries.

## Repository Structure
- `agents.py`: Defines individual agents responsible for rewriting query, retrieval, etc.
- `workflow.py`: Implements the LangGraph workflow, orchestrating the interaction between agents.
- `app.py`: Contains the Streamlit frontend, allowing users to input queries and receive AI-generated summaries.
- `config.py`: Contains API keys 
- `Evaluate_RAGs.pdf`: Provides an analysis and discussion of model outputs, evaluating retrieval effectiveness and summarization quality. (PDF)
- `Evaluate_RAGs.ipynb`: Provides an analysis and discussion of model outputs, evaluating retrieval effectiveness and summarization quality. (IPYNB for running)
- lora (directory) : contains fine-tuned lora models and files outlining code to train lora model.
- basic_rag (directory): Files to test out basic retreival RAG

**Download the lora folder from this link and place it in the root directory for code to work:** https://drive.google.com/drive/folders/1Ls-zU29i7Os2UI1HzoEd8w7LokwMxUMr?usp=sharing

## Running the code
### **User Interface**
To launch the Streamlit frontend, run:

```bash
streamlit run app.py
```
This will open a web interface where users can input job-related queries and receive AI-generated summaries.
### Evaluating RAG Systems
Run all the cells in `Evaluate_RAGs.ipynb` to see different RAG systems working along with interpretation.
```bash
jupyter notebook Evaluate_RAGs.ipynb
```
  
## Running the Unit Test
All the code runs under 5 minutes so there is no need for a unit test


