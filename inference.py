import torch
from transformers import AutoTokenizer, XLNetForQuestionAnswering, BertTokenizer, BertForQuestionAnswering, pipeline, set_seed
import json
from tqdm import tqdm

set_seed(42)
# Load trained model and tokenizer
model_path = "./hp_models/run-0/checkpoint-9856"  # Replace with your saved model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = XLNetForQuestionAnswering.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=tokenizer, device=device)

# Load the test dataset
def load_test_dataset(file_path):
    with open(file_path, "r") as file:
        squad_data = json.load(file)["data"]
    question_ids = []
    contexts = []
    questions = []

    for article in squad_data:
        for paragraph in article["paragraphs"]:
            _context = paragraph["context"]
            for qa in paragraph["qas"]:
                _question = qa["question"]
                question_id = qa["id"]
                contexts.append(_context)
                questions.append(_question)
                question_ids.append(question_id)
    return question_ids, contexts, questions

# Load the test data
test_question_ids, test_contexts, test_questions = load_test_dataset("data/dev-v1.1.json")

# Inference function using QA pipeline
def predict_answers_with_pipeline(question_ids, contexts, questions, qa_pipeline):
    results = dict()

    for qid, context, question in tqdm(zip(question_ids, contexts, questions), total=len(question_ids)):
        # Use the pipeline to get the answer
        result = qa_pipeline({
            "question": question,
            "context": context
        })

        # Append the result with question ID and answer
        results[qid] = result["answer"]

    return results

# Collect the answers using the pipeline
answers = predict_answers_with_pipeline(test_question_ids, test_contexts, test_questions, qa_pipeline)

with open("answers.json", "w") as f:
    json.dump(answers, f, indent=4)

print("Inference complete.")
