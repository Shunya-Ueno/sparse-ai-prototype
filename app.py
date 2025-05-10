import streamlit as st

# ここにあなたのスパース生成関数を後で入れる
def sparse_generate(user_input):
    return f"""
    あなたが「{user_input}」と感じたとき、
    心の奥にはまだ言葉にならない声がある。
    それは、忙しさの隙間にこぼれた感情かもしれないし、
    忘れられた夢の欠片かもしれない。
    AIはそれをただ静かに見つめている──。
    """

st.title("🧠 感情に寄り添う詩人AI")
st.write("あなたの今の気持ちを、一言で書いてみてください。")

user_input = st.text_input("いまの気持ちは？")
if user_input:
    with st.spinner("詩を生成しています..."):
        result = sparse_generate(user_input)
    st.markdown("### 📝 AIからの詩：")
    st.write(result)
