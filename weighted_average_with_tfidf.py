# weighted_average_with_tfidf.py

import numpy as np
import joblib
from sklearn.decomposition import SparseCoder
import sentencepiece as spm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.special import softmax

# 1. 準備: モデルとデータの読み込み
embedding_matrix = np.load("embedding_matrix.npy")
dictionary = joblib.load("dict_components.pkl")
sp = spm.SentencePieceProcessor()
sp.load("sp.model")

# 2. TF-IDF Vectorizer の準備（corpus.txt を使って SentencePiece トークンで学習）
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus_lines = [line.strip() for line in f.readlines() if line.strip()]

corpus_tokenized = [" ".join(sp.encode(line, out_type=str)) for line in corpus_lines]
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
tfidf_vectorizer.fit(corpus_tokenized)

# αを自動調整する関数（案A+B、範囲調整版）
def estimate_alpha_hybrid(prompt, tfidf_weights):
    length = len(sp.encode(prompt))  # トークン長
    tfidf_sum = tfidf_weights.sum()  # 情報量

    # 個別のα（安定化のため分母+1やε付き）
    alpha_length = 1.0 / (length + 1)
    alpha_tfidf = 1.0 / (tfidf_sum + 1e-8)

    # 重み付き平均（例：長さに0.4、TF-IDFに0.6の重み）
    hybrid_alpha = 0.4 * alpha_length + 0.6 * alpha_tfidf

    # 安定化のための範囲制限（例: 0.2〜0.8）
    return max(0.2, min(0.8, hybrid_alpha))

# 4. TF-IDF加重平均ベクトルによる予測関数（動的α対応）
def predict_next_token_tfidf(prompt):
    tokens = sp.encode(prompt, out_type=str)
    ids = sp.encode(prompt, out_type=int)

    if len(ids) == 0:
        return "<empty input>"

    vecs = embedding_matrix[ids]
    joined = " ".join(tokens)
    tfidf_weights = tfidf_vectorizer.transform([joined]).toarray().flatten()
    tfidf_weights = tfidf_weights[:len(vecs)]

    # sum が 0 なら mean で代替
    if tfidf_weights.sum() == 0:
        mean_vec = vecs.mean(axis=0).reshape(1, -1)
    else:
        tfidf_weights = tfidf_weights / (tfidf_weights.sum() + 1e-8)
        mean_vec = np.average(vecs, axis=0, weights=tfidf_weights).reshape(1, -1)

    # α を推定してスパースコーダを初期化
    alpha = estimate_alpha_hybrid(prompt, tfidf_weights)
    coder = SparseCoder(dictionary=dictionary, transform_algorithm='lasso_lars', transform_alpha=alpha)
    code = coder.transform(mean_vec)

    recon = np.dot(code, dictionary)
    similarities = np.dot(embedding_matrix, recon.T).squeeze()
    next_id = np.argmax(similarities)

    # softmax確率（案B）
    probs = softmax(similarities)
    confidence = probs[next_id]

    # cosine正規化スコア（案A）
    normed_sim = (similarities[next_id] - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)

    # ランキング（案C）
    rank = similarities.argsort()[::-1].tolist().index(next_id) + 1

    # 📋 複合評価ログ出力
    print(f"  α used: {alpha:.4f}")
    print(f"  Cosine similarity (raw): {similarities[next_id]:.4f}")
    print(f"  Cosine similarity (normalized): {normed_sim:.4f}")
    print(f"  Confidence (softmax): {confidence:.4f}")
    print(f"  Token rank: {rank}")
    top5_ids = similarities.argsort()[-5:][::-1]
    top5_tokens = [sp.id_to_piece(int(i)) for i in top5_ids]
    print(f"  Top-5 similar tokens: {top5_tokens}")

    return sp.id_to_piece(int(next_id))

# 5. テストプロンプト
def run_tests():
    prompts = [
        "The future of AI is",
        "Sparse modeling is",
        "I want to build",
        "My favorite language is",
        "Deep learning and sparse coding are"
    ]
    for prompt in prompts:
        print(f"Input: {prompt}")
        print("Next token:", predict_next_token_tfidf(prompt))
        print("---")

if __name__ == "__main__":
    run_tests()
