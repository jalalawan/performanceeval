# -*- coding: utf-8 -*-
"""
Created on Sat May 27 21:07:45 2023
@author: jawan
"""
import openai
import os
import streamlit as st
import spacy
from PyPDF2 import PdfFileReader
from docx import Document



# Set the path to the SSL certificate file
ssl_certificate_path = "C:/Users/awan/cacert.pem"

# Set the REQUESTS_CA_BUNDLE environment variable to the SSL certificate path
os.environ["REQUESTS_CA_BUNDLE"] = ssl_certificate_path

openai.api_key='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'

# Load English tokenizer
nlp = spacy.load("en_core_web_sm")


def text_to_chunks(text):
    chunks = []
    chunk = ""
    chunk_length = 0

    sentences = list(nlp(text).sents)

    for sentence in sentences:
        if chunk_length + len(sentence.text) > 16000:
            chunks.append(chunk)
            chunk = sentence.text
            chunk_length = len(sentence.text)
        else:
            chunk += " " + sentence.text
            chunk_length += len(sentence.text)

    if chunk:
        chunks.append(chunk)

    return chunks


def generate_summarizer(
    max_tokens,
    temperature,
    top_p,
    frequency_penalty,
    text,
    person_type,
):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for text summarization. Your responses are succinct and to the point.",
            },
            {
                "role": "user",
                "content": f"Summarize this for a {person_type}: {text}",
            },
        ],
    )
    return res["choices"][0]["message"]["content"]


def generate_answer(
    max_tokens,
    temperature,
    top_p,
    frequency_penalty,
    text,
    questions,
):
    # Initialize an empty list to store the answers
    answers = []

    # Start conversation with the text
    conversation = [
        {
            "role": "system",
            "content": "You are a knowledgeable assistant that can answer questions based on given text. You respond based on provided text and answer each question after thoroughly going through the provided information.",
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    # Process each question one by one
    for question in questions:
        conversation.append({"role": "user", "content": question})
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            messages=conversation,
        )
        # Append each answer to the list
        answers.append(res["choices"][0]["message"]["content"])
        # Append the assistant's response to the conversation
        conversation.append({"role": "assistant", "content": res["choices"][0]["message"]["content"]})

    return answers


def trim_output(output):
    sentences = output.split('. ')
    last_complete_sentence_index = len(sentences) - 1
    if not sentences[-1].endswith('.'):
        last_complete_sentence_index -= 1
    return '. '.join(sentences[:last_complete_sentence_index + 1]) + '.'


def chunk_text_and_process(
    max_tokens,
    temperature,
    top_p,
    frequency_penalty,
    text,
    questions,
    person_type,
    action,
):
    chunks = text_to_chunks(text)
    summary = ""
    answers = []
    
    for chunk in chunks:
        if action == "Summarize":
            result = generate_summarizer(max_tokens, temperature, top_p, frequency_penalty, chunk, person_type)
            summary += trim_output(result)
        elif action == "Answer the question" or action == "Both":
            results = generate_answer(max_tokens, temperature, top_p, frequency_penalty, chunk, questions)
            # Now trim each answer and store them in a list
            for result in results:
                answers.append(trim_output(result))

        if action == "Both":
            result = generate_summarizer(max_tokens, temperature, top_p, frequency_penalty, chunk, person_type)
            summary += trim_output(result)

    return summary, answers


def main():
    # Initialize the app
    st.title("QueryWhiz® Summarization and Question Answering")

    # Step 1: Input Text or Upload File
    st.header("Step 1: Input Text or Upload File")
    text_input_method = st.radio("Text input method", ('Type text', 'Upload file'))
    text = ""
    if text_input_method == 'Type text':
        text = st.text_area("Input Text")
    else:
        uploaded_files = st.file_uploader("Upload file(s)", type=['docx', 'pdf'], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.type == 'application/pdf':
                    pdfReader = PdfFileReader(uploaded_file)
                    text += " ".join(page.extract_text() for page in pdfReader.pages)
                else:  # file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    document = Document(uploaded_file)
                    text += " ".join([p.text for p in document.paragraphs])

    # Step 2: Choose your desired action
    st.header("Step 2: Choose your desired action")
    action = st.selectbox(
        "Choose the action",
        ("Summarize", "Answer the question", "Both"),
        index=2,
    )

    with st.expander("Current Parameters (Suggest Keeping Defaults)", expanded=False):
        max_tokens = st.slider("Token (Number of words the model should consider for each output)", min_value=10, max_value=500, value=100, step=10, key='max_tokens_slider')
        temperature = st.slider("Temperature (Higher values generate more diverse outputs, lower values more deterministic)", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key='temperature_slider')
        top_p = st.slider("Sampling top_p (Controls diversity of text using top_x%)", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key='top_p_slider')
        frequency_penalty = st.slider("Frequency Penalty (Penalizes new tokens based on their frequency)", min_value=-1.0, max_value=1.0, value=0.0, step=0.1, key='frequency_penalty_slider')

    # Step 3: Choose your style
    st.header("Step 3: Choose your style")
    person_type = st.selectbox(
        "Summarize the text for:",
        ("Scholar", "Data Scientist", "Expert", "Second Grader"),
        index=0,
    )

    # Step 4: Input Question (if user selected "Question Answering" or "Both")
    if action == "Answer the question" or action == "Both":
        st.header("Step 4: Input Question(s)")
        questions = [st.text_input(f"Input question {i + 1}") for i in range(4)]

    if st.button("Execute"):
        with st.spinner("Processing..."):
            # Remove empty questions before passing to the function
            valid_questions = [q for q in questions if q]
            summary, answers = chunk_text_and_process(
                max_tokens,
                temperature,
                top_p,
                frequency_penalty,
                text,
                valid_questions,
                person_type,
                action,
            )
        st.success("Processing complete!")

        if action == "Summarize" or action == "Both":
            st.header("Summarization")
            st.write(summary)

        if action == "Answer the question" or action == "Both":
            st.header("Answer(s)")
            for i, answer in enumerate(answers):
                # We map the answer to the corresponding non-empty question
                st.subheader(f"Answer to question {questions.index(valid_questions[i])+1}")
                st.write(answer)

    st.markdown('---')
    st.markdown('Developed by Jalal Awan. For feedback, please reach out via email at mawan@usc.edu or twitter @jalal_awan')

if __name__ == "__main__":
    main()