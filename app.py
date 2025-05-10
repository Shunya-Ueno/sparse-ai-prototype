# app.py
import streamlit as st
from sparse_coding import generate_sparse_output  # ← いま作った関数を呼び出す

st.title("🧠 感情に寄り添う詩人AI")
st.write("あなたの今の気持ちを、一言で書いてみてください。")

user_input = st.text_input("いまの気持ちは？")
if user_input:
    with st.spinner("詩を生成しています..."):
        result = generate_sparse_output(user_input, alpha=0.5)
    st.markdown("### 📝 AIからの詩：")
    st.write(result)
