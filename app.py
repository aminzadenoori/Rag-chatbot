import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from trubrics.integrations.streamlit import FeedbackCollector
from langchain_community.llms import HuggingFacePipeline
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import torch
from sentence_transformers import SentenceTransformer, util
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoModel
import atexit
import glob

HFtoken="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"

global llm
global user_question_temp

import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Initialize the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.getvalue().decode("utf-8") + "\n"
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=256,
        chunk_overlap=32,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    out_text = ""
    for chunk in chunks:
        out_text += chunk + "\n\n\n\n"
    with open("chunks.txt", "w") as file:
        file.write(out_text)
    return chunks

def get_vectorstore(text_chunks):

    model_name = "hkunlp/instructor-base"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
    # embeddings = HuggingFaceInstructEmbeddings(query_instruction="Represent the query for retrieval: ")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, option):
    st.session_state.chat_history = []
    llm = None
    if option == "xgen-7b-8k-base":
        llm = HuggingFaceHub(repo_id="Salesforce/xgen-7b-8k-base", model_kwargs={"temperature": 0.2, "max_length": 250}, huggingfacehub_api_token=HFtoken)
    elif option == "falcon-7b-instruct":
        llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.4, "max_length": 570}, huggingfacehub_api_token=HFtoken)
    elif option == "Mistral-7B-Instruct-v0.3":
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", model_kwargs={"temperature": 0.2,"max_new_tokens": 500,"max_new_length":500}, huggingfacehub_api_token=HFtoken)
    elif option == "Meta-Llama-3.1-8B-Instruct":
        llm = HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct", model_kwargs={"temperature": 0.2,"max_new_tokens": 500,"max_new_length":500, "stop": ["\n\n", "Context:", "Question:", "Answer:"]}, huggingfacehub_api_token=HFtoken)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    custom_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    Use the information below to answer the user's question as accurately as possible. Provide only the answer without additional commentary or repetition. If the answer is not clear from the context, simply state "I don't know."
    {context}
    
    Question:
    {question}

    Helpful Answer:
    """
)
    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever(k=2),prompt=custom_prompt)
    return qa_chain

def handle_userinput(user_question):
    response = st.session_state.conversation.run(user_question)
    feedback_collector = FeedbackCollector()
    feedback_results = []
    feedback = []
    print(response)
    helpful_answer_index = response.find('Helpful Answer:')
    if helpful_answer_index != -1:
        generated_answer = response[helpful_answer_index + len('Helpful Answer:'):].strip()
    st.write(bot_template.replace("{{MSG}}", generated_answer), unsafe_allow_html=True)
    feedback.append(feedback_collector)
    st.session_state.feedback_results = feedback

# Function to calculate ROUGE score
def calculate_rogue(generated_answer, expected_answer):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(expected_answer, generated_answer)
    return scores['rougeL'].fmeasure

# Function to calculate BLEU score
def calculate_blue(generated_answer, expected_answer):
    reference = expected_answer.split()
    hypothesis = generated_answer.split()
    return sentence_bleu([reference], hypothesis)


def display_feedback():
    if "feedback_results" in st.session_state:
        st.subheader("Collected Feedback")
        for i, feedback in enumerate(st.session_state.feedback_results):
            st.write(f"Feedback {i+1}: {feedback.dict}")
    else:
        st.write("No feedback collected yet.")

def chat_history_to_txt():
    output = ""
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            output += message.content + ":\n"
        else:
            output += "\t\t" + message.content.strip().split('\n\n')[0] + "\n\n"
    with open("/Users/amin/ask-multiple-pdfs/thumbs_feedback.json", 'r') as file:
        data = json.load(file)
    user_response = data.get('user_response', None)
    output += "\n\n" + str(user_response)
    return output

def extract_context(text):
    # Find the index of "Context" and "Question"
    context_start = text.find("Context:")
    question_start = text.find("Question:")
    
    # Extract the text between "Context" and "Question"
    if context_start != -1 and question_start != -1:
        return text[context_start + len("Context:"):question_start].strip()
    else:
        return "Either 'Context' or 'Question' not found in the text."

def get_next_index():
    if os.path.exists('counter.txt'):
        with open('counter.txt', 'r') as f:
            index = int(f.read())
    else:
        index = 0
    index += 1
    with open('counter.txt', 'w') as f:
        f.write(str(index))
    return index

def collect_feedback_to_csv():
    json_files = glob.glob('thumbs_feedback*.json')
    data_list = []

    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            user_response = data.get('user_response', {})
            question = data.get('question', '')
            user_satisfaction = user_response.get('User satisfaction: faces', '')

            data_list.append({
                'Question': question,
                'User Satisfaction': user_satisfaction
            })

    df = pd.DataFrame(data_list)
    df.to_csv('feedback_results.csv', index=False)

atexit.register(collect_feedback_to_csv)
def main():
    load_dotenv()
    st.set_page_config(page_title="The First ChatBot (Beta)", page_icon=":robot:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Virtual Assistant  :robot_face:")
    user_question = st.text_input("Ask a question about your documents:")
    if st.button("Answer"):
        handle_userinput(user_question)
    # Get the next index
    i = get_next_index()
    feedback_filename = f"thumbs_feedback{i}.json"
    
    feedback_collector = FeedbackCollector()
    feedback_collector.st_feedback(feedback_type="faces", path=feedback_filename)

    # Open and read the thumbs_feedback.json file
    # Check if the file exists
    if os.path.exists(feedback_filename):
    # Open and read the thumbs_feedback.json file

    # Initialize data if the file doesn't exist
        data = []
        with open(feedback_filename, 'r') as json_file:
            data = json.load(json_file)



        # Add the "question" key to the data
        data["question"] = user_question

    # Write the updated data back to the JSON file
        with open(feedback_filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    if st.session_state.chat_history:
        csv_data = chat_history_to_txt()
        st.download_button("Download Chat History as CSV", csv_data, "results.txt", "text/csv")
    with st.sidebar:
        st.subheader("Your documents")
        uploaded_docs = st.file_uploader("Upload your documents here and click on 'Process'", accept_multiple_files=True, type=['pdf', 'txt'])
        option = st.selectbox('Select a Large language model:', ["Mistral-7B-Instruct-v0.3","Meta-Llama-3.1-8B-Instruct"])
        if st.button("Process"):
            with st.spinner("Processing"):
                pdf_docs = [doc for doc in uploaded_docs if doc.type == "application/pdf"]
                txt_docs = [doc for doc in uploaded_docs if doc.type == "text/plain"]
                raw_text = ""
                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)
                if txt_docs:
                    raw_text += get_txt_text(txt_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.option = option
                st.session_state.conversation = get_conversation_chain(vectorstore, option)
        st.subheader("Load Questions CSV")
        questions_file = st.file_uploader("Upload a CSV file with questions", type=["csv"])
        if questions_file is not None:
            questions_df = pd.read_csv(questions_file)
            st.dataframe(questions_df)
        if st.button("Generate Answers"):
            generated_answers = []
            automatic_scores = []
            rogue_scores = []
            blue_scores = []
            context=[]
            
            for question, expected_answer in zip(questions_df["Question"], questions_df["Expected Answer"]):
                response = st.session_state.conversation.run(question)
                

                context.append(extract_context(response))
                helpful_answer_index = response.find('Helpful Answer:')
                if helpful_answer_index != -1:
                    generated_answer = response[helpful_answer_index + len('Helpful Answer:'):].strip()
                else:
                    generated_answer = response
                generated_answers.append(generated_answer)
                
                # # Calculate similarity score
                # embeddings1 = model.encode(generated_answer, convert_to_tensor=True)
                # embeddings2 = model.encode(expected_answer, convert_to_tensor=True)
                # similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()
                # automatic_scores.append(similarity_score)
                
                # # Calculate ROUGE score
                # rogue_score = calculate_rogue(generated_answer, expected_answer)
                # rogue_scores.append(rogue_score)
                
                # # Calculate BLEU score
                # blue_score = calculate_blue(generated_answer, expected_answer)
                # blue_scores.append(blue_score)

            questions_df['Generated answer by LLM'] = generated_answers
            questions_df['LLM'] = st.session_state.option
            # questions_df['Automatic Score'] = automatic_scores
            # questions_df['ROUGE Score'] = rogue_scores
            # questions_df['BLEU Score'] = blue_scores
            questions_df['Context'] = context
            st.write("Answers Generated:")
            st.dataframe(questions_df)
            csv_data = questions_df.to_csv(index=False)
            st.download_button("Download Results", csv_data, "results_with_answers.csv", "text/csv")

if __name__ == '__main__':
    main()
