import streamlit as st

# 👉 TEM QUE VIR PRIMEIRO
st.set_page_config(page_title="Chat ComunicaBR", page_icon="🇧🇷")

from PIL import Image
from responder import responder

# Mostra imagem
caminho_imagem = "governo-federal-brasil.png"
imagem = Image.open(caminho_imagem)
st.image(imagem, width=200)

# Título e descrição
st.title("💬 Chat IA - ComunicaBR")
st.markdown("Faça uma pergunta sobre os dados dos relatórios governamentais por estado ou Brasil.")

# Input
pergunta = st.text_input("Digite sua pergunta:")

if st.button("Perguntar") and pergunta.strip():
    with st.spinner("Consultando os dados..."):
        try:
            resposta = responder(pergunta)
            st.success("✅ Resposta gerada!")
            st.markdown(resposta)
        except Exception as e:
            st.error(f"Erro: {e}")