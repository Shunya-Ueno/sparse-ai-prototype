import urllib.request
import zipfile
import os

url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
out_path = "wikitext-2.zip"

# ダウンロード
urllib.request.urlretrieve(url, out_path)

# 解凍
with zipfile.ZipFile(out_path, 'r') as zip_ref:
    zip_ref.extractall("wikitext-2")

print("✅ WikiText-2 downloaded and extracted.")
