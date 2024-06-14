import os
import time
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS



# Load environment variables
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# Function to extract text from PDF
def get_pdf_text(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=10000, 
    chunk_overlap=1000,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# def analyse_paper(prompt_text):
#     # Display the progress bar with initial text
#     progress_text = "Analysis in progress. Please wait."
#     my_bar = st.progress(0, text=progress_text)

#     # Simulate work by incrementally updating the progress bar
#     for percent_complete in range(100):
#         time.sleep(0.10)  # Sleep to simulate the analysis process
#         my_bar.progress(percent_complete + 1, text=progress_text)
    
#     # Once the analysis is done, empty the progress bar
#     my_bar.empty()

#     # Placeholder for the GenerativeModel code
#     # Uncomment and modify with actual implementation
#     model = genai.GenerativeModel(model_name="gemini-1.5-flash")  # generation_config=generation_config)
#     response = model.generate_content(prompt_text)
    
#     return response
# Define the Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("Risk of Bias Analyzer")

    # Sidebar for file uploader
    with st.sidebar:
        st.header("Upload Your Research Paper")
        uploaded_file = st.file_uploader("Upload Research Paper", type=["pdf"])

    st.markdown("""
    <style>
        .main-content {
            padding: 2px;
            background-color: #f9f9f9;
            border-radius: 10px;
        }
        .footer {
            padding: 2px;
            background-color: #f9f9f9;
            border-radius: 10px;
            text-align: center;
            margin-top: 100px;
            font-size: 0.8em;
            color: #888;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.header("Get Detailed Risk of Bias Analysis")

    st.markdown("""
    Welcome to the Risk of Bias Analyzer! Using the power of **Gemini AI**, we provide a detailed risk of bias analysis for each domain of your research paper. Simply upload your PDF and click 'Analyse' to get started.
    """)

    if uploaded_file is not None:
        # Get the text from the PDF
        pdf_text = get_pdf_text(uploaded_file)
        text_chunks = get_text_chunks(pdf_text)
        
        # Create vector store
        vector_store = get_vector_store(text_chunks) 

        # Construct prompt
        prompt_parts = [
            "You are an expert scientific researcher with extensive experience in conducting systematic literature surveys and meta-analyses. "
            "Your task is to analyze the following research paper using the Revised Cochrane risk-of-bias tool for randomized trials (RoB 2)"
            "Please provide detailed answers to the following questions for each domain of bias:\n\n"
            "Research paper:\n"
            f"{pdf_text}\n\n"
            "Domain 1: Risk of bias arising from the randomization process\n"
            "1.1 Was the allocation sequence random?\n"
            "1.2 Was the allocation sequence concealed until participants were enrolled and assigned to interventions?\n"
            "1.3 Did baseline differences between intervention groups suggest a problem with the randomization process?\n"
            "1.4 Risk-of-bias judgement\n"
            "1.5 Optional: What is the predicted direction of bias arising from the randomization process?\n\n"
            "Domain 2: Risk of bias due to deviations from the intended interventions (effect of assignment to intervention)\n"
            "2.1 Were participants aware of their assigned intervention during the trial?\n"
            "2.2 Were carers and people delivering the interventions aware of participants' assigned intervention during the trial?\n"
            "2.3 If Y/PY/NI to 2.1 or 2.2: Were there deviations from the intended intervention that arose because of the trial context?\n"
            "2.4 If Y/PY to 2.3: Were these deviations likely to have affected the outcome?\n"
            "2.5 If Y/PY/NI to 2.4: Were these deviations from intended intervention balanced between groups?\n"
            "2.6 Was an appropriate analysis used to estimate the effect of assignment to intervention?\n"
            "2.7 If N/PN/NI to 2.6: Was there potential for a substantial impact (on the result) of the failure to analyse participants in the group to which they were randomized?\n"
            "2.8 Risk-of-bias judgement\n"
            "2.9 Optional: What is the predicted direction of bias due to deviations from intended interventions?\n\n"
            "Domain 2.1: Risk of bias due to deviations from the intended interventions (effect of adhering to intervention)\n"
            "2.1.1 Were participants aware of their assigned intervention during the trial?\n"
            "2.1.2 Were carers and people delivering the interventions aware of participants' assigned intervention during the trial?\n"
            "2.1.3 [If applicable:] If Y/PY/NI to 2.1.1 or 2.1.2: Were important non-protocol interventions balanced across intervention groups?\n"
            "2.1.4 [If applicable:] Were there failures in implementing the intervention that could have affected the outcome?\n"
            "2.1.5 [If applicable:] Was there non-adherence to the assigned intervention regimen that could have affected participants’ outcomes?\n"
            "2.1.6 If N/PN/NI to 2.1.3, or Y/PY/NI to 2.1.4 or 2.1.5: Was an appropriate analysis used to estimate the effect of adhering to the intervention?\n"
            "2.1.7 Risk-of-bias judgement\n"
            "2.1.8 Optional: What is the predicted direction of bias due to deviations from intended interventions?\n\n"
            "Domain 3: Risk of bias due to missing outcome data\n"
            "3.1 Were data for this outcome available for all, or nearly all, participants randomized?\n"
            "3.2 If N/PN/NI to 3.1: Is there evidence that the result was not biased by missing outcome data?\n"
            "3.3 If N/PN to 3.2: Could missingness in the outcome depend on its true value?\n"
            "3.4 If Y/PY/NI to 3.3: Is it likely that missingness in the outcome depended on its true value?\n"
            "3.5 Risk-of-bias judgement\n"
            "3.6 Optional: What is the predicted direction of bias due to missing outcome data?\n\n"
            "Domain 4: Risk of bias in measurement of the outcome\n"
            "4.1 Was the method of measuring the outcome inappropriate?\n"
            "4.2 Could measurement or ascertainment of the outcome have differed between intervention groups?\n"
            "4.3 If N/PN/NI to 4.1 and 4.2: Were outcome assessors aware of the intervention received by study participants?\n"
            "4.4 If Y/PY/NI to 4.3: Could assessment of the outcome have been influenced by knowledge of intervention received?\n"
            "4.5 If Y/PY/NI to 4.4: Is it likely that assessment of the outcome was influenced by knowledge of intervention received?\n"
            "4.6 Risk-of-bias judgement\n"
            "4.7 Optional: What is the predicted direction of bias in measurement of the outcome?\n\n"
            "Domain 5: Risk of bias in selection of the reported result\n"
            "5.1 Were the data that produced this result analysed in accordance with a pre-specified analysis plan that was finalized before unblinded outcome data were available for analysis?\n"
            "5.2 Is the numerical result being assessed likely to have been selected, on the basis of the results, from multiple eligible outcome measurements (e.g. scales, definitions, time points) within the outcome domain?\n"
            "5.3 Is the numerical result being assessed likely to have been selected, on the basis of the results, from multiple eligible analyses of the data?\n"
            "5.4 Risk-of-bias judgement\n"
            "5.5 Optional: What is the predicted direction of bias due to selection of the reported result?\n\n"
        ]

        # Concatenate the prompt parts into a single string
        prompt_text = ''.join(prompt_parts)

        # Button to summarize
        if st.button("Analyse"):
            try:
                with st.spinner("Analyzing..."):
                    generation_config = {
                        "candidate_count": 1,
                        "max_output_tokens": 90000,
                        "temperature": 1.0,
                        "top_p": 0.7,
                    }

                    safety_settings=[
                    {
                        "category": "HARM_CATEGORY_DANGEROUS",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE",
                    },
                    ]
                    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config,
                                                        safety_settings=safety_settings)
                    response = model.generate_content(prompt_text)

                # Display the response
                st.markdown("## Summary of Research Paper")
                st.markdown(response.text)
                print(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
            st.markdown("---")
            st.markdown("**Note:** This summary was generated by an AI model and may vary in the answers provided.")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
