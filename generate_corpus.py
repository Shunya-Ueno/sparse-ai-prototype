from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_data = dataset["train"]

with open("corpus.txt", "w", encoding="utf-8") as f:
    for example in train_data:
        line = example["text"].strip()
        if line and not line.startswith("="):
            f.write(line + "\n")

print("✅ corpus.txt が作成されました")
