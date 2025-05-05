# dictionary_learning.py

import numpy as np
import joblib
from sklearn.decomposition import MiniBatchDictionaryLearning
import logging

# ログ設定
logging.basicConfig(
    filename='dictionary_learning.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

logging.info("✅ 辞書学習 開始")

# 事前に保存された埋め込み行列をロード（build_embeddings.pyで保存したやつ）
embeddings = np.load("embedding_matrix.npy")

# MiniBatchDictionaryLearning モデル定義
dict_learner = MiniBatchDictionaryLearning(
    n_components=4096,  # 基底の数
    alpha=1.0,
    batch_size=64,
    max_iter=20000,
    random_state=42,
    verbose=True,
    tol=1e-6
)

# 学習
dictionary = dict_learner.fit(embeddings).components_
logging.info("✅ 辞書学習 完了。shape: %s", dictionary.shape)

# 保存
np.save("dictionary.npy", dictionary)
joblib.dump(dict_learner, "dictionary_model.pkl")

logging.info("✅ ファイル保存完了")
print("✅ 辞書学習完了:", dictionary.shape)

# SparseCoder 用に components_ だけも別保存
joblib.dump(dictionary, "dict_components.pkl")
print("✅ 辞書を dict_components.pkl に保存しました")

