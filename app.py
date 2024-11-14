import json
import sys
from datetime import datetime
import random
import optuna
import pandas as pd
import torch
#from sklearn.model_selection import train_test_split
import evaluate
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BertForQuestionAnswering,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    XLNetForQuestionAnsweringSimple,
    pipeline,
    set_seed,
)
import numpy as np

SEED = 42
MODEL_NAME = "xlnet/xlnet-large-cased"
# MODEL_NAME = "bert-base-cased"
BERT_MAX_LENGTH = 512
set_seed(SEED)
random.seed(SEED)
assert torch.cuda.is_available()
device = torch.device("cuda")
# Load pre-trained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
)  # use AutoTokenizer, lest return_offset errs


def model_init():
    return XLNetForQuestionAnsweringSimple.from_pretrained(MODEL_NAME)


# Load SQuAD dataset (from json file)
def load_dataset(file_path, is_mini=False):
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
    samples = list(zip(contexts, questions, answers))
    random.shuffle(samples)
    contexts, questions, answers = zip(*samples)
    contexts = list(contexts)
    questions = list(questions)
    answers = list(answers)
    
    if is_mini:
        mini_size = 1000
        return contexts[:mini_size], questions[:mini_size], answers[:mini_size]
    return contexts, questions, answers


class SquadDataset(Dataset):
    def __init__(self, contexts, questions, answers, tokenizer):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.encodings = self._preprocess()

    def _preprocess(self):
        encodings = self.tokenizer(  # TODO: Maybe excessive padding tokens?
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


contexts_train, questions_train, answers_train = load_dataset("data/train-v1.1.json")
contexts_test, questions_test, answers_test = load_dataset("data/dev-v1.1.json", is_mini=True)
train_dataset = SquadDataset(contexts_train, questions_train, answers_train, tokenizer)
test_dataset = SquadDataset(contexts_test, questions_test, answers_test, tokenizer)
data_collator = DefaultDataCollator()  # padding already done


#metric = evaluate.load("squad")
#def compute_metrics(start_logits, end_logits, features, examples):
#    n_best_logits = 5
#    example_to_features = collections.defaultdict(list)
#    for idx, feature in enumerate(features):
#        example_to_features[feature["example_id"]].append(idx)
#
#    predicted_answers = []
#    for example in tqdm(examples):
#        example_id = example["id"]
#        context = example["context"]
#        answers = []
#
#        # Loop through all features associated with that example
#        for feature_index in example_to_features[example_id]:
#            start_logit = start_logits[feature_index]
#            end_logit = end_logits[feature_index]
#            offsets = features[feature_index]["offset_mapping"]
#
#            start_indexes = np.argsort(start_logit)[-1 : -n_best_logits - 1 : -1].tolist()
#            end_indexes = np.argsort(end_logit)[-1 : -n_best_logits - 1 : -1].tolist()
#            for start_index in start_indexes:
#                for end_index in end_indexes:
#                    # Skip answers that are not fully in the context
#                    if offsets[start_index] is None or offsets[end_index] is None:
#                        continue
#                    # Skip answers with a length that is either < 0 or > max_answer_length
#                    if (
#                        end_index < start_index
#                    ):
#                        continue
#
#                    answer = {
#                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
#                        "logit_score": start_logit[start_index] + end_logit[end_index],
#                    }
#                    answers.append(answer)
#
#        # Select the answer with the best score
#        if len(answers) > 0:
#            best_answer = max(answers, key=lambda x: x["logit_score"])
#            predicted_answers.append(
#                {"id": example_id, "prediction_text": best_answer["text"]}
#            )
#        else:
#            predicted_answers.append({"id": example_id, "prediction_text": ""})
#
#    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
#    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

training_args = TrainingArguments(
    # output_dir="./models",
    output_dir=f"./hp_models/{datetime.now().strftime('%d-%m_%H-%M')}",
    #learning_rate=4.457e-5,
    bf16=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    #gradient_accumulation_steps=4,
    #num_train_epochs=1,
    #weight_decay=0.150,
    #warmup_steps=178,
    logging_dir=f"./logs/{datetime.now().strftime('%d-%m_%H-%M')}",
    logging_steps=100,
    save_total_limit=2,
    report_to="tensorboard",
    eval_strategy="epoch",
    save_strategy="epoch",
    no_cuda=False
)

trainer = Trainer(
    model=None,
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics
)


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        # "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [64, 128]),
        "gradient_accumulation_steps": trial.suggest_categorical("per_device_train_batch_size", [1, 2, 4]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.2),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2,4),
    }


best_trials = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=10,
)

#trainer.train()

