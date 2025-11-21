from datasets import load_dataset

dataset = load_dataset("opus_books", "en-fr")
print(dataset.keys())

print("Nombre d'exemples dans le train :", len(dataset["train"]))

print("\n--- Exemples ---")
for i in range(3):
    example = dataset["train"][i]
    print(f"\nExemple {i+1}:")
    print(f"EN: {example['translation']['en']}")
    print(f"FR: {example['translation']['fr']}")
