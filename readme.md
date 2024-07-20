# A RAG-driven ChatBot Based on Open Source LLMs

![ChatBot App Diagram](./docs/d-W3SuscVT.svg)

## Introduction
------------
This application, known as the ChatBot, leverages open-source Large Language Models (LLMs) to enable interactive conversations with multiple PDF documents. It employs advanced natural language processing techniques, allowing users to pose questions about the contents of PDFs and receive precise responses. The ChatBot incorporates various open-source language models to deliver accurate and efficient answers.

## How It Works
------------
The ChatBot operates through the following steps:
1. PDF Loading: The application loads multiple PDF documents and extracts their text content.
2. Text Chunking: The extracted text is divided into smaller, manageable chunks to enhance processing efficiency.
3. Language Model Integration: Text chunks are transformed into vector representations using open-source embedding techniques.
4. Similarity Matching: User queries are compared against the text chunks to find the most relevant information.
5. Response Generation: Based on the relevant text chunks, language models generate responses to user queries.
6. Response Evaluation: Users can evaluate the quality of the generated responses using a "faces" scoring system, providing feedback on the effectiveness of the ChatBot's answers.

## Technical Details
------------
The ChatBot uses the Streamlit framework for the frontend and integrates several open-source backend technologies:
- Text Extraction and Processing: Utilizes PyPDF2, a free tool for handling PDF files.
- Language Models and Embeddings: Employs HuggingFaceInstructEmbeddings and models from the HuggingFace Hub (google/flan-t5-large, Salesforce/xgen-7b-8k-base), all of which are open-source projects.
- Vector Storage and Retrieval: Uses FAISS, an open-source library for efficient similarity search and clustering of dense vectors.
- Interactive UI: Streamlit provides an interactive web interface, allowing users to upload documents, input questions, and receive responses.
- Feedback Mechanism: Incorporates a feedback system where users can score the responses using a simple and intuitive "faces" rating system, helping to gather insights on user satisfaction and ChatBot performance.

## Docker Container Support
------------
The ChatBot can be deployed as a Docker container, making it easy to set up and run in any environment that supports Docker. A Dockerfile is included in the repository, which details the steps for creating a Docker image that runs the ChatBot.
### Building the Docker Image
To build the Docker image, run the following command in your terminal where the Dockerfile is located:
``` docker build -t chatbot . ```
This command builds a Docker image named chatbot based on the instructions in the Dockerfile.
### Running the Docker Container
To run the ChatBot as a Docker container, use the following command:
``` docker run -p 8501:8501 chatbot ```
This command runs the chatbot Docker container and maps port 8501 of the container to port 8501 on your host, allowing you to access the Streamlit app by navigating to http://localhost:8501 in your web browser.

## Chat History Saving Feature
------------
Users have the option to save their chat history as a CSV file, which includes the questions asked, the responses provided by the ChatBot, and the feedback given by the users. This feature facilitates easy tracking and review of interactions, making it a valuable tool for user experience improvement and future enhancements.

## Dependencies and Installation
------------
To install the ChatBot, please follow these steps:
1. Clone the repository to your local machine.
2. Install the required dependencies by running the following command:
   pip install -r requirements.txt
3. Obtain an API key from OpenAI and add it to the .env file in the project directory.
   HUGGINGFACEHUB_API_TOKEN=your_secret_api_key

## Usage
-----
To use the ChatBot, follow these steps:
1. Ensure that you have installed the required dependencies and added the OpenAI API key to the .env file.
2. Run the application using the Streamlit command:
   streamlit run app.py
3. Follow the in-app instructions to upload your PDF documents and interact with the chat interface.
4. Provide feedback on the responses you receive by using the "faces" scoring system, directly influencing future improvements and updates.
5. Save your chat history to a CSV file for future reference or analysis.


## License
-------
The ChatBot is released under the MIT License.
