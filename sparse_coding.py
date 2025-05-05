# sparse_coding.py
import numpy as np
import joblib
from sklearn.decomposition import SparseCoder
import sentencepiece as spm

# 1. 必要なデータを読み込み
embedding_matrix = np.load("embedding_matrix.npy")
dictionary = joblib.load("dict_components.pkl")
sp = spm.SentencePieceProcessor()
sp.load("sp.model")

# 3. 平均スパースコードを使った次トークン予測関数（coderを引数にする）
def predict_next_token(tokens, coder, top_n=1):
    ids = sp.encode(tokens, out_type=int)
    vecs = embedding_matrix[ids]
    mean_vec = vecs.mean(axis=0).reshape(1, -1)
    code = coder.transform(mean_vec)
    recon = np.dot(code, dictionary)
    similarities = np.dot(embedding_matrix, recon.T).squeeze()
    next_id = int(np.argmax(similarities))
    return sp.id_to_piece(next_id)

# --- テスト用プロンプト（いくつか実験） ---
prompts = [
    "The future of AI is",
    "Sparse modeling is",
    "I want to build",
    "My favorite language is",
    "Deep learning and sparse coding are",
    "命とは",
    "Thank you"
]

# 追加：transform_alpha を変えて評価
alphas = [0.1, 0.3, 0.5, 0.7, 1.0]

for alpha in alphas:
    print(f"\n===== α = {alpha} =====")
    coder = SparseCoder(dictionary=dictionary, transform_algorithm='lasso_lars', transform_alpha=alpha)
    for prompt in prompts:
        try:
            print(f"Input: {prompt}")
            print("Next token:", predict_next_token(prompt, coder))
        except Exception as e:
            print("Error:", e)
