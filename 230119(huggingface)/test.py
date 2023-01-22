# HuggingFace Transformers를 활용한 문장 분류 모델 학습

import random
import logging

import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset, load_metric, ClassLabel, Sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

model_checkpoint = "klue/roberta-base"
batch_size = 128
task = "nli" # Task : Natural language Inference

datasets = load_dataset("klue", task)
# print(datasets["train"][0])

metric = load_metric("glue", "qnli")
fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
# print(fake_preds, fake_labels)

test = metric.compute(predictions=fake_preds, references=fake_labels)
# print(test)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
token_test = tokenizer("힛걸 진심 최고로 멋지다.", "힛걸 진심 최고다 그 어떤 히어로보다 멋지다")
# print(token_test)

sentence1_key, sentence2_key = {"premise", "hypothesis"}
# print(f"Sentence 1 : {datasets['train'][0][sentence1_key]}")

def preprocess_function(examples):
    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        truncation=True,
        return_token_type_ids=False
    )

encoded_datasets = datasets.map(preprocess_function, batched=True)

# Load SequenceClassification Model
num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

def compute_metrics(eval_pred) : 
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

metric_name = "accuracy"

args = TrainingArguments(
    "test-nli",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model=metric_name
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
metrics = trainer.evaluate(encoded_datasets["validation"])
print(metrics)