import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, XLNetForQuestionAnswering, BertTokenizer, BertForQuestionAnswering, pipeline, set_seed
import json
from tqdm import tqdm

set_seed(42)
# Load trained model and tokenizer
#model_path = "./hp_models/12-11_23-46/checkpoint-912"
paths = [
"./hp_models/13-11_16-10/run-0/checkpoint-1368",
"./hp_models/13-11_16-10/run-1/checkpoint-684",
"./hp_models/13-11_21-16/run-0/checkpoint-1369",
"./hp_models/13-11_21-16/run-0/checkpoint-2052",
"./hp_models/13-11_21-16/run-1/checkpoint-4107",
"./hp_models/13-11_21-16/run-1/checkpoint-5476",
"./hp_models/13-11_21-16/run-2/checkpoint-5476",
"./hp_models/13-11_21-16/run-3/checkpoint-2738",
"./hp_models/13-11_21-16/run-4/checkpoint-2736"
]
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
        result = qa_pipeline(question=question, context=context)

        # Append the result with question ID and answer
        results[qid] = result["answer"]

    return results

for i, model_path in enumerate(paths):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_MAX_LENGTH = 512 
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, max_length=MODEL_MAX_LENGTH, device=device)

    answers = predict_answers_with_pipeline(test_question_ids, test_contexts, test_questions, qa_pipeline)
    
    
    with open(f"answers/trial{i}.json", "w") as f:
        json.dump(answers, f, indent=4)

print("Inference complete.")
