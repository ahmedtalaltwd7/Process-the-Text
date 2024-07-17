import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Step 1: Load the Documents
def load_documents(directory):
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            try:
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                    documents[filename] = file.read()
            except UnicodeDecodeError:
                with open(os.path.join(directory, filename), 'r', encoding='utf-8', errors='ignore') as file:
                    documents[filename] = file.read()
    return documents

# Step 2: Load the Model
def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    return tokenizer, model

# Step 3: Process the Text and Answer Questions
def answer_questions(documents, questions, tokenizer, model):
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    results = []
    for doc_name, content in documents.items():
        for question in questions:
            answer = qa_pipeline({'question': question, 'context': content})
            results.append({'Document': doc_name, 'Question': question, 'Answer': answer['answer']})
    return results

# Example Usage
if __name__ == "__main__":
    # Specify the directory containing documents and the model directory
    docs_directory = 'doc'
    model_directory = 'mod'  # Path to the mod folder

    # Load documents
    documents = load_documents(docs_directory)

    # Load the model
    tokenizer, model = load_model(model_directory)

    # List of questions to answer
    questions = [
        "What is the main topic of the document?",
        "Who is the author?",
        "What are the key findings?"
    ]

    # Get answers
    answers = answer_questions(documents, questions, tokenizer, model)

    # Convert answers to DataFrame for better readability
    answers_df = pd.DataFrame(answers)
    print(answers_df)
