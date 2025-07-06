# Question-Answering System for Geoportale Nazionale Archeologia (GNA)

This repository contains the codebase and evaluation material for the design and development of a **Question-Answering (QA) system** for the [Geoportale Nazionale per l’Archeologia (GNA)](https://gna.cultura.gov.it). The project was carried out as part of the internship in preparation for my Master’s thesis in <i>[Digital Humanities and Digital Knowledge](https://corsi.unibo.it/2cycle/DigitalHumanitiesKnowledge)</i> at the University of Bologna. 

<br>

### Project Overview

The GNA QA system is a retrieval-augmented question-answering assistant designed to respond to natural language queries based on official GNA documentation. It integrates web crawling, document chunking, vector-based retrieval and generation in a modular and scalable architecture.
<br><br>


> ### The application is available at:  
> ### 👉 [gna-assistant-ai.streamlit.app](https://gna-assistant-ai.streamlit.app/)

<br>

### **Main features include:**

- Focused crawling and sitemap generation from GNA wiki operative manual
- Chunked document processing and metadata annotation  
- Dense embeddings generation using multilingual Sentence Transformers  
- Retrieval-augmented generation
- Citation-aware prompting  
- Streamlit-based user interface and feedback tracking 
- Evaluation suite for retrieval metrics (Precision@k, Recall@k, MRR)
<br><br>


## 📂 Structure

- `generate_sitemap.py`, `create_chunks.py`, `vector_store.py`: Knowledge base preparation  
- `rag_sys.py`: Retrieval-Augmented Generation pipeline  
- `main.py`: Streamlit application logic  
- `feedback_handling.py`: Feedback management
- `evaluate_retrieval.py`: Evaluation framework  
- `create_test_data.py`: Test set generation  
- `main_preprocess.py`: Combined pipeline for sitemap, chunking, and vectorization  
- `data/`: Document chunks, test datasets, metrics, logs  
- `feedback/`: Local SQLite database for user feedback  
- `sitemap/`: XML sitemap of the GNA website  
- `OCR/`: OCR-related scripts
- `.faiss_db/`: FAISS vector store  
- `.streamlit/`: Streamlit configuration files  
- `requirements.txt`: Python dependencies  
- `packages.txt`: Additional system requirements for Streamlit Cloud
<br><br>


### 📊 Evaluation

Automated evaluation was performed using a synthetic test set of 400 domain-specific questions. 
Key metrics include:

- **Precision@5**
- **Recall@5**
- **MRR (Mean-Reciprocal Rank)**
- **Avg. Retrieval Time**


## Acknowledgments

This project was supervised and supported by **Mario Caruso** and **Simone Persiani** from [BUP Solutions](https://www.bupsolutions.com/en/home_en/), whose guidance and technical insights were instrumental throughout the internship. I sincerely thank them for their time, encouragement, and valuable mentorship.
