import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

model_name = "t5-small"
batch_size = 8
epochs = 3
learning_rate = 2e-5
max_length = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Utilisation de : {device}")

print("Chargement du modèle T5...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

print("Chargement du dataset OPUS Books...")
dataset = load_dataset("opus_books", "en-fr")

train_size = min(10000, len(dataset["train"]))
val_size = min(1000, len(dataset["train"]))

print(f"Train: {train_size} exemples")
print(f"Validation: {val_size} exemples")

train_data = dataset["train"].select(range(train_size))
val_data = dataset["train"].select(range(train_size, train_size + val_size))

def preprocess_function(examples):
    prefix = "translate English to French: "
    inputs = [prefix + ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("\nTokenisation des données...")
tokenized_train = train_data.map(
    preprocess_function,
    batched=True,
    remove_columns=train_data.column_names,
    num_proc=1
)
tokenized_val = val_data.map(
    preprocess_function,
    batched=True,
    remove_columns=val_data.column_names,
    num_proc=1
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./modele_traduction_checkpoints",
    eval_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    warmup_steps=500,
    dataloader_num_workers=0,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("\nDébut de l'entraînement...")
trainer.train()

print("\nÉvaluation finale...")
results = trainer.evaluate()
print(f"Résultats: {results}")

save_path = "./modele_traduction"
print(f"\nSauvegarde du modèle dans '{save_path}/'...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

config = {
    "model_name": model_name,
    "max_length": max_length,
    "source_lang": "en",
    "target_lang": "fr",
    "task_prefix": "translate English to French: "
}
torch.save(config, f"{save_path}/config.pth")

print(f"Modèle sauvegardé dans '{save_path}/'")
print("Entraînement terminé !")
