# Semantic Support Ticket Retriever (RAG Prototype)

This project implements a **Retrieval-Augmented Generation (RAG)** system for retrieving contextual answers from historic support tickets. It uses **FAISS** for similarity search and a transformer-based model for generating semantic embeddings.

## Description

This application allows users to input a support query and retrieve the most relevant historical support tickets. It uses semantic search to ensure contextual relevance and is built with **Streamlit**. Users can also provide feedback on the retrieved results to improve the system.

## Features

- Retrieve top-k similar tickets based on a user's support query.
- Semantic search using transformer-based sentence embeddings.
- Feedback system for improving search relevance.

## Installation

### Prerequisites

- Python 3.7 or later
- `pip` (Python package installer)

### Step-by-step Guide

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Tanvi-Sharmaaa/RAG-system-backend.git
    cd RAG-system-backend
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv rag-env
    ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     .\rag-env\Scripts\activate
     ```

4. **Install dependencies**:
    ```bash
    pip install streamlit faiss-cpu sentence-transformers
    ```

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Access the application**:
    - After running the app, open the link provided in the terminal (usually `http://localhost:8501`) to interact with the UI.

## Technologies Used

- **Streamlit**: For building the interactive web application UI quickly and efficiently.
- **FAISS**: Used for fast and scalable similarity search across vector embeddings. It's highly optimized for performance and handles large-scale retrieval well.
- **Sentence-Transformers**: Enables generating dense semantic embeddings. Specifically, we use the `intfloat/multilingual-e5-large-instruct` model, which:
  - Is optimized for **instruction-following** tasks.
  - Supports **multilingual queries**, making it robust for diverse users.
  - Generates **high-quality embeddings** suitable for semantic search, improving relevance and accuracy in retrieval tasks.

