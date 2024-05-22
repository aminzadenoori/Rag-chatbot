import os
import glob
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain import LLMChain
from langchain import PromptTemplate
from trubrics.integrations.streamlit import FeedbackCollector
from transformers import AutoModel
from transformers import AutoModelForCausalLM
import csv
import io
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import json
import pandas as pd

global llm
global user_question_temp

def get_pdf_text(pdf_docs):
    text = ""
    # docs_dir='docs'
    # # Navigate through each subdirectory in the 'docs' directory
    # for subdir in os.listdir(docs_dir):
    #     subdir_path = os.path.join(docs_dir, subdir)
    #     # Check if 'English' folder exists in the current subdirectory
    #     english_folder_path = os.path.join(subdir_path, 'en')
    #     if os.path.exists(english_folder_path):
    #         # Find all PDF files in the 'English' folder
    #         pdf_files = glob.glob(os.path.join(english_folder_path, '*.pdf'))
    #         # Read each PDF file
    #         for pdf_file in pdf_files:
    #             pdf_reader = PdfReader(pdf_file)
    #             for page in pdf_reader.pages:
    #                 text += page.extract_text() + "\n"
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
# Function to get text from TXT files
def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        # Ensure the file is read in the correct mode
       
        text += txt.getvalue().decode("utf-8") + "\n"
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=32,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # Open file in write mode
    out_text=""
    for chunck in chunks:
        out_text += chunck + "\n\n\n\n"

    with open("chuncks.txt", "w") as file:
    # Write text to the file
        file.write(out_text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    # Load the model
    model = AutoModel.from_pretrained("hkunlp/instructor-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings
)
    return vectorstore


def get_conversation_chain(vectorstore,option):
    
    # llm = ChatOpenAI()
    st.session_state.chat_history=[]
    llm = None
    if option=="xgen-7b-8k-base":
        llm = HuggingFaceHub(repo_id="Salesforce/xgen-7b-8k-base",model_kwargs={"temperature":0.2, "max_length":250},huggingfacehub_api_token="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")
    if option=="Falcon-7b-instruct":
        llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct",model_kwargs={"temperature":0.5, "max_length":570},huggingfacehub_api_token="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")
    if option=="google/flan-t5-large":
        llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.2, "max_length":570},huggingfacehub_api_token="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")
    if option=="phi3":
        llm = HuggingFaceHub(repo_id="microsoft/Phi-3-mini-4k-instruct", model_kwargs={"trust_remote_code":True,"temperature":0.2, "max_length":570},huggingfacehub_api_token="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")
    #         model = AutoModelForCausalLM.from_pretrained(
#         "microsoft/Phi-3-mini-128k-instruct",
#         trust_remote_code=True, 

# )
#         tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

#         pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512
# )
#         llm= HuggingFacePipeline(pipeline=pipe)

    print(llm("what was the first disney movie?"))
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(),
    #     memory=memory
    # )
    qa_chain = RetrievalQA.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(k=3)
)
  
    
    return qa_chain


def handle_userinput(user_question):
    #response = st.session_state.conversation({'question': user_question})
    response = st.session_state.conversation({'query': user_question})
   
    print(response)

    
    feedback_collector = FeedbackCollector()
    feedback_results = []
    feedback=[]
    
    helpful_answer_index = response.find('Helpful Answer:')
    if helpful_answer_index != -1:
                    generated_answer = response[helpful_answer_index + len('Helpful Answer:'):].strip()
                    
    st.write(bot_template.replace(
                "{{MSG}}",  generated_answer), unsafe_allow_html=True)
           
    feedback.append(feedback_collector)
    print(feedback)
        # Store feedback results in session state
    st.session_state.feedback_results = feedback

def display_feedback():
    if "feedback_results" in st.session_state:
        st.subheader("Collected Feedback")
        print(st.session_state.feedback_results)
        for i, feedback in enumerate(st.session_state.feedback_results):
            st.write(f"Feedback {i+1}: {feedback.dict}")
    else:
        st.write("No feedback collected yet.")

def chat_history_to_txt():
    """Convert chat history to CSV format, alternating between question and answer."""
    output=""
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            output += message.content + ":\n"  # User messages as questions
        else:
            output += "\t\t" + message.content.strip().split('\n\n')[0] + "\n\n"  # Bot responses as answers
    
    with open("/Users/amin/ask-multiple-pdfs/thumbs_feedback.json", 'r') as file:
        data = json.load(file)        
    # Load the full JSON data from the file
    
    # Access and return the 'user_response' field
    user_response = data.get('user_response', None)
    output += "\n\n" + str(user_response)  # Bot responses as answers

    return output

def main():
    load_dotenv()
    st.set_page_config(page_title="The First ChatBot (Beta)",
                       page_icon=":robot:")
    st.write(css, unsafe_allow_html=True)
    
    
    # if st.button("Display Feedback"):
    #     display_feedback()
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Virtual Assistant  :robot_face:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if st.button("Answer"):
       handle_userinput(user_question)
    feedback_collector = FeedbackCollector()
    feedback_collector.st_feedback(feedback_type="faces" 
	,path="thumbs_feedback.json")
    if st.session_state.chat_history:
        csv_data = chat_history_to_txt()
        st.download_button("Download Chat History as CSV", csv_data, "results.txt", "text/csv")
    

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_docs = st.file_uploader(
            "Upload your documents here and click on 'Process'", accept_multiple_files=True, type=['pdf', 'txt'])
        option = st.selectbox(
        'Select a Large langaue model:',
        ['falcon-7b-instruct', 'xgen-7b-8k-base', 'google/flan-t5-large',"phi3"]
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # Separate PDF and TXT files based on their extension
                pdf_docs = [doc for doc in uploaded_docs if doc.type == "application/pdf"]
                txt_docs = [doc for doc in uploaded_docs if doc.type == "text/plain"]
                
                # Initialize raw_text
                raw_text = ""
                
                # Process PDF files if any
                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)
                
                # Process TXT files if any
                if txt_docs:
                    raw_text += get_txt_text(txt_docs)


                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks) 
                st.session_state.vectorstore = vectorstore  # Store vectorstore in session state
                st.session_state.option = option  # Store option in session state


                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore,option)
        st.subheader("Load Questions CSV")
        questions_file = st.file_uploader("Upload a CSV file with questions", type=["csv"])
        
        if questions_file is not None:
            questions_df = pd.read_csv(questions_file)
            st.dataframe(questions_df)
            
            if st.button("Generate Answers"):
                 # Initialize list to store generated answers
                generated_answers = []
                
                print(questions_df.columns)

                for question in questions_df["Question"]:
                    # Reset the conversation chain for each question
                    response = st.session_state.conversation.run(question)
                    # Extract the final answer from the response
                    helpful_answer_index = response.find('Helpful Answer:')
                    if helpful_answer_index != -1:
                        generated_answer = response[helpful_answer_index + len('Helpful Answer:'):].strip()
                    print(generated_answer)
                    generated_answers.append(generated_answer)

                questions_df['Generated answer by LLM'] = generated_answers
                questions_df['LLM'] = st.session_state.option
                
                
                st.write("Answers Generated:")
                st.dataframe(questions_df)

                csv_data = questions_df.to_csv(index=False)
                st.download_button("Download Results", csv_data, "results_with_answers.csv", "text/csv")

                
   
if __name__ == '__main__':
    main()
