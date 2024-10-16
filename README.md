# SmartMail AI - AI-Powered Email Classification and Response System

## Overview

**SmartMail AI** is an AI-based solution for managing email classification and response generation for university departments, particularly for Heads of Departments (HODs). This system automates the classification of emails and generates responses based on document repository content, leveraging **fine-tuned BERT**, **Chroma DB** for document retrieval, and **Llama 3** for detecting sensitive information and generating responses.

---

## Process Overview

### 1. **Data Generation and Model Fine-tuning**

1. **Data Generation**:
   - For each email class (e.g., **Corporate Inquiry**, **Student Inquiry**, and **Academic Collaboration Inquiry**), 10 categories were defined.
   - Llama 3 was used via Groq to generate 30 emails for each category, making up a large dataset.

2. **Fine-tuning BERT**:
   - The dataset generated from Llama 3 was used to fine-tune the **BERT-large-uncased** model using Hugging Face’s `transformers` library. This fine-tuned model was saved as `Kunjjasoria/smartsense` on Hugging Face.

### 2. **Email Classification and Workflow**

- The application uses a fine-tuned **BERT** model to classify incoming emails into different categories:
  - **Corporate Inquiry**: Directly forwarded to the HOD.
  - **Non-Corporate Inquiry**: Processed further using document retrieval and response generation.

### 3. **Document Retrieval with Chroma DB**

- **Chroma DB** is used for retrieving relevant documents from the university’s repositories (e.g., syllabi, research papers) to provide context for generating email responses.
- Documents are uploaded using `chroma_upload.py`, which splits PDFs into smaller chunks and stores them in the Chroma database for efficient similarity search.

### 4. **Sensitive Content Detection and Response Generation with Llama 3**

- **Llama 3** is leveraged through the **Groq API** to detect whether an email contains sensitive information (e.g., confidential partnerships, legal matters). If sensitive content is detected, the email is flagged and forwarded to the HOD.
- If no sensitive content is found, Llama 3 generates a response to the email using the context retrieved from Chroma DB.

---

## Repository Structure

```
.
├── data/                     # Contains the Chroma DB embeddings and data files
├── src/                      # Source code for the application
│   ├── utils/                # Utility scripts
│   │   └── chroma_upload.py  # Script for uploading PDFs to Chroma DB
│   └── app.py                # Main application file for the Streamlit chatbot
├── requirements.txt          # Required Python packages
└── README.md                 # This file
```

---

## Key Components

### 1. **Fine-tuned BERT Model for Email Classification**
   - The model `Kunjjasoria/smartsense` is loaded using Hugging Face’s `transformers` to classify emails into categories such as **Corporate Inquiry** or **Non-Corporate Inquiry**.
   
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def initialize_model():
    model_name = "Kunjjasoria/smartsense"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return classifier
```

### 2. **Document Retrieval Using Chroma DB**
   - Chroma DB is used to store and retrieve context from document repositories based on email content using a similarity search.
   - Documents (PDFs) are uploaded using the script `chroma_upload.py`.

```python
def add_files(path_to_folder):
    for filepath in glob.glob(path_to_folder, recursive=True):
        if filepath.endswith('.pdf'):
            process_pdf_file(filepath)
```

### 3. **Llama 3 for Sensitive Content Detection and Response Generation**
   - Emails not classified as **Corporate Inquiry** are further processed by passing the content and retrieved documents to Llama 3 via the Groq API. Llama 3 detects sensitive content or drafts a response based on the context.

```python
def llama3_classify_and_respond(email, relevant_documents):
    system_prompt = "..."
    chat_completion = groq_chat.completions.create(...)
    return chat_completion.choices[0].message.content
```

---

## Running the Application

### 1. **Clone the Repository**

```bash
git clone https://github.com/Kunjjasoria/smartsense.git
cd smartsense
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Upload Documents to Chroma DB**

Before running the application, upload your document data to Chroma DB using `chroma_upload.py`:

```bash
python src/utils/chroma_upload.py
```

### 4. **Run the Streamlit App**

```bash
streamlit run src/app.py
```

Once the app is running, you can input an email subject and body, and the system will classify the email and either forward it to the HOD or generate a response using the context retrieved from the document repositories.

---

## How It Works

1. **Email Classification**: 
   - Input emails are classified as either **Corporate Inquiry** or **Non-Corporate Inquiry** using the fine-tuned BERT model.

2. **Context Retrieval**:
   - For non-corporate inquiries, the system uses Chroma DB to retrieve relevant context from stored documents using a similarity search.

3. **Sensitive Content Detection**:
   - Llama 3 processes the email and context to detect any sensitive information. If found, the email is flagged for forwarding to the HOD.

4. **Response Generation**:
   - If no sensitive content is found, Llama 3 drafts a response based on the email content and the retrieved documents.

---

## Future Enhancements

- **Multilingual Support**: Expand to handle emails in multiple languages.
- **Additional Document Formats**: Add support for a wider range of document formats (e.g., Word, Excel).
- **Improved Response Generation**: Enhance Llama 3’s capabilities for generating highly specific responses based on a broader context.

---

## License

This project is licensed under the MIT License.

---

This README explains the entire process of building the SmartSense application, from data generation, model fine-tuning, and document retrieval to email classification and response generation using Llama 3. Let me know if you need any adjustments!
