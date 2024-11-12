import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

OPENAI_API_KEY = "REPLACE THIS WITH AN OPENAI KEY, sk-..."
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

class ExtractiveQABatch:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.qa_chain = self.create_qa_chain()

    def create_qa_chain(self):
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant that answers questions based on the provided context."),
                ("system", "The answer should be a span from the context. Do not make up an answer or further process it, just extract the answer."),
                ("human", "{input}"),
            ]
        )
        return qa_prompt

    def generate_answer(self, question, context):
        input_text = f"Context: {context}\nQuestion: {question}"
        
        # Generating the response using the formatted input and prompt
        prompt = self.qa_chain.format(input=input_text)
        response = self.llm.generate([prompt])  # Pass the formatted prompt as a list
        
        return response.generations[0][0].text.strip()

def load_test_dataset(file_path):
    with open(file_path, "r") as file:
        squad_data = json.load(file)["data"]
    question_ids, contexts, questions = [], [], []

    for article in squad_data:
        for paragraph in article["paragraphs"]:
            _context = paragraph["context"]
            for qa in paragraph["qas"]:
                question_ids.append(qa["id"])
                contexts.append(_context)
                questions.append(qa["question"])
    return question_ids, contexts, questions

def predict_answers(question_ids, contexts, questions, qa_model):
    results = {}
    for qid, context, question in tqdm(zip(question_ids, contexts, questions), total=len(question_ids)):
        answer = qa_model.generate_answer(question, context)
        results[qid] = answer
    return results

def clean_answers(answers):
    # Strip fullstop from the end of the answer and "Answer:" from the start
    cleaned_answers = {qid: answer.strip("Answer:").strip().strip(".") for qid, answer in answers.items()}
    return cleaned_answers


if __name__ == '__main__':
    qa_model = ExtractiveQABatch()
    question_ids, contexts, questions = load_test_dataset("data/dev-v1.1.json")
    answers = predict_answers(question_ids, contexts, questions, qa_model)
    cleaned_answers = clean_answers(answers)

    with open("answers.json", "w") as f:
        json.dump(cleaned_answers, f, indent=4)