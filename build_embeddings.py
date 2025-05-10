# build_embeddings.py

import numpy as np
import sentencepiece as spm

# 1. GloVeベクトルの読み込み
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, encoding="utf8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# 2. SentencePiece モデルのロード
sp = spm.SentencePieceProcessor()
sp.load("sp.model")  # 事前にtrain_tokenizer.pyで生成済み

# 3. 語彙サイズと埋め込み次元
vocab_size = sp.get_piece_size()
embedding_dim = 50

# 4. 埋め込み行列を初期化
embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype='float32')

# 5. GloVe埋め込みのロード
glove_path = "glove.6B.50d.txt"
glove = load_glove_embeddings(glove_path)

# 6. トークン → 埋め込み に変換
for i in range(vocab_size):
    token = sp.id_to_piece(i)
    if token in glove:
        embedding_matrix[i] = glove[token]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

# 7. 保存（任意）
np.save("embedding_matrix.npy", embedding_matrix)

print(f"✅ 埋め込み行列の構築完了: {embedding_matrix.shape}")
