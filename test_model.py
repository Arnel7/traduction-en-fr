import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

config = torch.load("./modele_traduction/config.pth")
print(f"Config: {config}")

task_prefix = config.get("task_prefix", "translate English to French: ")

print("Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained("./modele_traduction")
model = AutoModelForSeq2SeqLM.from_pretrained("./modele_traduction")
model.eval()

def translate(texts, direction="en-fr", max_length=128):
    if isinstance(texts, str):
        texts = [texts]

    if direction == "en-fr":
        prefix = "translate English to French: "
    elif direction == "fr-en":
        prefix = "translate French to English: "
    else:
        raise ValueError("direction doit être 'en-fr' ou 'fr-en'")

    texts_with_prefix = [prefix + text for text in texts]

    inputs = tokenizer(
        texts_with_prefix,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    with torch.no_grad():
        translated = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translations

print("=" * 70)
print("TESTS DE TRADUCTION ANGLAIS → FRANÇAIS")
print("=" * 70 + "\n")

textes_en = [
    "Hello, how are you today?",
    "The weather is beautiful this morning.",
    "I love learning new languages.",
    "This book is very interesting.",
    "Can you help me please?"
]

translations_fr = translate(textes_en, direction="en-fr")

for en, fr in zip(textes_en, translations_fr):
    print(f"EN: {en}")
    print(f"FR: {fr}")
    print()

print("=" * 70)
print("TESTS DE TRADUCTION FRANÇAIS → ANGLAIS")
print("=" * 70 + "\n")

textes_fr = [
    "Bonjour, comment allez-vous aujourd'hui?",
    "Le temps est magnifique ce matin.",
    "J'adore apprendre de nouvelles langues.",
    "Ce livre est très intéressant.",
    "Pouvez-vous m'aider s'il vous plaît?"
]

translations_en = translate(textes_fr, direction="fr-en")

for fr, en in zip(textes_fr, translations_en):
    print(f"FR: {fr}")
    print(f"EN: {en}")
    print()
