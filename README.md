# Data Science Document Q&A
## About the Project

This project is a **Document Question-Answering** application powered by the **Gemma Model** and **Streamlit**. It allows users to upload documents, create a vector store for document embedding and ask questions directly related to the uploaded content. The answers are generated based on the context from the documents.

---

## **Table of Contents**
- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [File Structure](#File-Structure)
- [Troubleshooting](#Troubleshooting)



---

## Features

- **Interactive UI**: Built with Streamlit for easy interaction.
- **PDF Document Processing**: Load and process documents stored in the `./data` directory.
- **Vector Store Creation**: Generate vector embeddings for documents using **FAISS** and **Google Generative AI Embeddings**.
- **Advanced Question Answering**: Use the **Gemma Model (gemma2-9b-it)** to generate accurate, context-specific answers.
- **Document Similarity Search**: Display relevant document chunks used to generate the answer.
- **Real-Time Response Time**: Measure and display the time taken to generate a response.
- **Multi-Document Support**: Handles multiple documents seamlessly for broader context.
- **Content Summary Feature**: Generate summaries of document content to provide quick insights.

---

## Tech Stack

- **Python**
- **Streamlit**
- **Groq**
- **FAISS** (Facebook AI Similarity Search)
- **Google Generative AI** for embeddings
- **LangChain** for processing and chains
- **dotenv** for secure environment variable handling

---

## Model Architecture

1. **Input Layer**:
   - User uploads documents in PDF format.
   - User inputs a query through the Streamlit interface.

2. **Document Preprocessing**:
   - **PyPDFLoader** extracts text from the uploaded PDFs.
   - Text is split into chunks using **RecursiveCharacterTextSplitter** to ensure manageable sizes for embedding and retrieval.

3. **Vectorization**:
   - **Google Generative AI Embeddings** converts text chunks into high-dimensional vector representations.

4. **Vector Store**:
   - **FAISS** stores and indexes the vector embeddings for efficient similarity searches.

5. **Question Processing**:
   - User query is processed through a **ChatPromptTemplate** to guide the language model.

6. **Retriever**:
   - The retriever searches the FAISS vector store to find the most relevant document chunks for the query.

7. **LLM (Gemma Model)**:
   - Processes the retrieved document context and the query.
   - Generates a precise, context-specific answer.

8. **Output Layer**:
   - Displays the generated answer.
   - Shows relevant document chunks used in the similarity search.
  

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- `pip` (Python package installer)

### Steps to Run

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set Up a Virtual Environment**
   ```bash
   conda create -n myenv python=3.9 
   source myenv/bin/activate   # On Windows: myenv\Scripts\activate
   ```

3. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Environment Variables**
   - Create a `.env` file in the project root directory.
   - Add the following keys:
     ```
     GROQ_API_KEY=<your_groq_api_key>
     GOOGLE_API_KEY=<your_google_api_key>
     ```

5. **Prepare Data**
   - Place your PDF files inside the `./data` directory.

6. **Run the Application**
   ```bash
   streamlit run app.py
   ```

7. **Access the App**
   - Open your browser and go to [http://localhost:8501](http://localhost:8501).

---

## Usage

1. **Initialize Vector Store**:
   - Click the **"Create Vector Store"** button to prepare the vector database.

2. **Ask Questions**:
   - Type your question in the input field, e.g., "What is the purpose of this document?"
   - Press Enter to get an answer.

3. **View Similar Documents**:
   - Expand the **"Document Similarity Search"** section to view relevant document chunks used to generate the answer.

4. **Response Time**:
   - Monitor real-time response time displayed below the answer for performance insights.

---

## File Structure

```
|-- data/                     # Directory for storing PDF documents
|-- app.py                    # Main Streamlit application code
|-- requirements.txt          # Required dependencies
|-- .env                      # Environment variables 
```

---

## Troubleshooting

- **Vector Store Not Ready:**
  - Ensure you click the "Create Vector Store" button before asking questions.
- **API Key Errors:**
  - Verify that your `.env` file contains valid API keys.
- **Dependencies Issues:**
  - Run `pip install -r requirements.txt` to ensure all required packages are installed.
- **No Response Generated:**
  - Check if the uploaded documents are correctly processed.
  - Verify the document format and ensure the data folder is not empty.

---


  
