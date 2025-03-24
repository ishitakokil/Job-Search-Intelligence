# Advanced RAG (Retrieval Augmented Generation) System for Job Search Intelligence
This project showcases a multi-agent, Retrieval-Augmented Generation (RAG) pipeline designed to answer job search-related questions using a combination of document retrieval, fine-tuned summarization, and LLM-based reasoning. The system is evaluated across multiple configurations, from a base LLM to an advanced agentic RAG with a fine-tuned LoRA model.

## About the Data
The LinkedIn Job Postings dataset is a comprehensive collection of job listings sourced from LinkedIn. This dataset contains information about various job postings, including job titles, companies, locations, employment types, industries, and more. Link to dataset: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

## Toggleable LoRA Integration
The system is designed to allow dynamic workflow customization. Users can toggle between basic RAG and LoRA-enhanced RAG in both the backend workflow and the frontend UI. This flexibility enables comparisons between retrieval-only responses and LoRA fine-tuned summaries.

