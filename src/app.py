import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import RobertaModel
import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

groq_chat = ChatGroq(
            groq_api_key="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", 
            model_name="llama-3.1-70b-versatile",
    )
client1 = chromadb.PersistentClient(path=r"../data/chroma_db")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)
db = Chroma(client=client1, embedding_function=embeddings)
@st.cache_resource()
def initialize_model():
    model_name = "Kunjjasoria/smartsense"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return classifier

classifier = initialize_model()

def create_email_string(subject, body):
    return f"Subject: {subject}.\n{body}"
def llama3_classify_and_respond(email, relevant_documents):
    system_prompt2 = (
            "You are an AI model tasked with two responsibilities:"
            "1. **Detect Sensitive Information**: Identify if the email contains any sensitive information like **confidential partnerships**, **legal matters**, etc. "
            "If such sensitive content is found, your response should be: 'Email will be forwarded to the HOD.'"
            "2. **Drafting Responses**: If no sensitive information is found, use the provided **context** (relevant documents) to help you draft an appropriate response to the email."
            "3. Your response should either be: 'Email will be forwarded to the HOD.' or draft a response based on the email and relevant documents, providing the necessary information."
            "4. Dont mention anything in the response that is not present in the email or relevant documents."
        )
    prompt2 = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt2),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    conversation2 = LLMChain(
        llm=groq_chat,
        prompt=prompt2,
        verbose=False,
    )

    output = conversation2.predict(
        human_input=f"Email Content:\n{email}\n\nRelevant Information:\n{relevant_documents}"
    )
    return output

def classify_and_handle_email(classifier, subject, body):
    email_string = create_email_string(subject, body)
    classification_result = classifier(email_string)[0]['label']
    if classification_result == "Corporate Inquiry":
        return "Mail will be forwarded to the HOD"
    else:
        relevant_documents = db.similarity_search(email_string)
        response = llama3_classify_and_respond(email_string, relevant_documents)
        return response
st.title("Email Assistant Chatbot")

if 'history' not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    st.write(message)

with st.form("email_form"):
    subject = st.text_input("Email Subject")
    body = st.text_area("Email Body")
    submitted = st.form_submit_button("Submit")

if submitted:
    result = classify_and_handle_email(classifier, subject, body)
    st.session_state.history.append(f"**User**: Subject: {subject}\n{body}")
    st.session_state.history.append(f"**AI**: {result}")
    st.query_params(rerun="true")