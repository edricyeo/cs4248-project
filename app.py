from transformers import BertForQuestionAnswering, BertTokenizer, Trainer, TrainingArguments, pipeline
import torch
import json
from torch.utils.data import Dataset

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Load SQuAD dataset (from json file)
def load_dataset(file_path):
    with open(file_path, "r") as file:
        squad_data = json.load(file)["data"]
    contexts = []
    questions = []
    start_positions = []
    end_positions = []
    
    for article in squad_data:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answers"][0]  # currently only use the first answer TODO: need to update this 
                contexts.append(context)
                questions.append(question)
                
                # Calculate start and end positions
                start_position = context.index(answer["text"])
                end_position = start_position + len(answer["text"]) - 1
                start_positions.append(start_position)
                end_positions.append(end_position)
    
    return contexts, questions, start_positions, end_positions

train_contexts, train_questions, train_start_positions, train_end_positions = load_dataset("data/train-v1.1.json")

# Custom dataset class
class SquadDataset(Dataset):
    def __init__(self, contexts, questions, start_positions, end_positions):
        self.contexts = contexts
        self.questions = questions
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.questions[idx],
            self.contexts[idx],
            max_length=384,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        item = {
            'input_ids': encodings['input_ids'].flatten(),
            'attention_mask': encodings['attention_mask'].flatten(),
            'start_positions': torch.tensor(self.start_positions[idx], dtype=torch.long),
            'end_positions': torch.tensor(self.end_positions[idx], dtype=torch.long),
        }
        return item

# Create dataset
train_dataset = SquadDataset(train_contexts, train_questions, train_start_positions, train_end_positions)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./models",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Pipeline and function to answer questions
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def answer_question(question: str, context: str):
    return qa_pipeline(question=question, context=context)

# Basic CLI for getting questions and answers
if __name__ == "__main__":
    context = input("Please enter a context: \n")
    while True:
        question = input("Please enter a question: \n")
        if question == "exit()":
            print("Goodbye!")
            break
        result = answer_question(question, context)
        print(f"Answer: {result['answer']}")