import os

import datasets
import evaluate
import numpy as np
import pandas as pd
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":

    metric = evaluate.load("accuracy")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    df = pd.read_table(f'ch06/train.txt', header=None)
    df.columns = ["text", "label"]
    valid = pd.read_table(f'ch06/valid.txt', header=None)
    valid.columns = ["text", "label"]
    test = pd.read_table(f'ch06/test.txt', header=None)
    test.columns = ["text", "label"]

    train_dataset = datasets.Dataset.from_pandas(df[["text", "label"]])
    train_tokenized = train_dataset.map(preprocess_function, batched=True)
    val_dataset = datasets.Dataset.from_pandas(valid[["text", "label"]])
    val_tokenized = val_dataset.map(preprocess_function, batched=True)
    test_dataset = datasets.Dataset.from_pandas(test[["text"]])
    test_tokenized = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=4
    )

    training_args = TrainingArguments(
        output_dir=f"./results",
        learning_rate=4e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=250,
        load_best_model_at_end=True,
        save_steps=1000,
        gradient_accumulation_steps=3,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    oof_results = trainer.predict(test_dataset=val_tokenized)
    np.save(f"oof_prediction", oof_results.predictions)

    results = trainer.predict(test_dataset=test_tokenized)
    np.save(f"test_prediction", results.predictions)
