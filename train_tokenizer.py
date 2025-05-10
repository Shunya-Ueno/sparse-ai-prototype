import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="sp",
    vocab_size=8000,
    model_type="bpe"
)

print("✅ トークナイザー学習完了。sp.model と sp.vocab が生成されました。")
