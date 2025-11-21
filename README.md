# Traduction Anglais → Français avec T5

## Description

Système de traduction automatique basé sur **T5 (Text-to-Text Transfer Transformer)**.

T5 traite toutes les tâches NLP comme des tâches texte-vers-texte en utilisant un **préfixe de tâche**.

## Structure

```
translation/
├── train.py        # Entraînement du modèle
├── test_model.py   # Tests de traduction
├── charger.py      # Chargement du dataset
└── README.md
```

## Installation

```bash
source venv/bin/activate
pip install torch transformers datasets
```

## Utilisation

### 1. Entraîner le modèle

```bash
python train.py
```

- Modèle : `t5-small`
- Dataset : OPUS Books (EN→FR)
- 10K exemples d'entraînement, 1K validation
- Préfixe : `"translate English to French: "`

### 2. Tester le modèle

```bash
python test_model.py
```

Tests EN→FR et FR→EN (FR→EN ne fonctionne pas car non entraîné).

### 3. Utiliser dans votre code

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("./modele_traduction")
model = AutoModelForSeq2SeqLM.from_pretrained("./modele_traduction")

def translate(text):
    input_text = "translate English to French: " + text
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translate("Hello, how are you?"))
```

## Configuration

```python
model_name = "t5-small"
batch_size = 8
epochs = 3
learning_rate = 2e-5
max_length = 128
```

## Préfixes T5

| Tâche | Préfixe |
|-------|---------|
| EN→FR | `translate English to French:` |
| FR→EN | `translate French to English:` |
| Résumé | `summarize:` |

**Important** : T5 nécessite obligatoirement un préfixe pour savoir quelle tâche effectuer.

## Ressources

- [Documentation T5](https://huggingface.co/docs/transformers/model_doc/t5)
- [OPUS Books Dataset](https://opus.nlpl.eu/opus-100.php)
# traduction-en-fr
