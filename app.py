import json

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BertForQuestionAnswering,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)

SEED = 42
MODEL_NAME = "bert-base-cased"
BERT_MAX_LENGTH = 512
set_seed(SEED)
assert torch.cuda.is_available()
device = torch.device("cuda")
# Load pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)  # use AutoTokenizer, lest return_offset errs
model = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)


# Load SQuAD dataset (from json file)
def load_dataset(file_path):
    with open(file_path, "r") as file:
        squad_data = json.load(file)["data"]
    contexts = []
    questions = []
    answers = []

    for article in squad_data:
        for paragraph in article["paragraphs"]:
            _context = paragraph["context"]
            for qa in paragraph["qas"]:
                _question = qa["question"]
                _answer = qa["answers"][0]  # all train samples only 1 answer
                contexts.append(_context)
                questions.append(_question)
                answers.append(_answer)
    return contexts, questions, answers


class SquadDataset(Dataset):
    def __init__(self, contexts, questions, answers, tokenizer):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.encodings = self._preprocess()

    def _preprocess(self):
        encodings = self.tokenizer(
            self.questions,
            self.contexts,
            max_length=BERT_MAX_LENGTH,  # TODO: Maybe shrink
            truncation="only_second",  # Ensure only the context is truncated
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,  # Get offset mapping for answer span
        )

        start_positions = []
        end_positions = []
        for i, answer in enumerate(self.answers):
            start_char_idx = answer["answer_start"]
            end_char_idx = start_char_idx + len(answer["text"])

            offsets = encodings["offset_mapping"][i]

            # Find the token indices that match the start and end character positions
            sequence_ids = encodings.sequence_ids(
                i
            )  # https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/question_answering.ipynb#scrollTo=fmh8ORIa8yA_
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if (
                offsets[context_start][0] > end_char_idx
                or offsets[context_end][1] < start_char_idx
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offsets[idx][0] <= start_char_idx:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offsets[idx][1] >= end_char_idx:
                    idx -= 1
                end_positions.append(idx + 1)

        encodings["start_positions"] = start_positions
        encodings["end_positions"] = end_positions
        encodings.pop("offset_mapping")  # not used anymore
        return encodings

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "start_positions": self.encodings["start_positions"][idx],
            "end_positions": self.encodings["end_positions"][idx],
        }


flattend_dataset = load_dataset("data/train-v1.1.json")
(
    contexts_train,
    contexts_test,
    questions_train,
    questions_test,
    answers_train,
    answers_test,
) = train_test_split(*flattend_dataset, test_size=0.2, random_state=SEED)
train_dataset = SquadDataset(contexts_train, questions_train, answers_train, tokenizer)
test_dataset = SquadDataset(contexts_test, questions_test, answers_test, tokenizer)
data_collator = DefaultDataCollator()  # padding already done

training_args = TrainingArguments(
    output_dir="./models",
    learning_rate=2e-5,
    bf16=True,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

model.save_pretrained("./fine-tuned-bert-qa")

# Pipeline and function to answer questions
qa_pipeline = pipeline(
    "question-answering", model=model, tokenizer=tokenizer, device=device
)


def answer_question(question: str, context: str):
    return qa_pipeline(question=question, context=context)

