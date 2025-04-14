# -------------------------
# 📙 IMPORTAÇÕES
# -------------------------
import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate  

# -------------------------
# 🔑 VARIÁVEIS DE AMBIENTE
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# -------------------------
# 📁 CAMINHOS
# -------------------------
CAMINHO_JSON = "chunks_comunicabr_2.json"


# -------------------------
# 🧍‍ CHUNKS EM OBJETOS DOCUMENT
# -------------------------
with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
    dados = json.load(f)

# Converte os chunks em objetos Document
documentos = [
    Document(page_content=chunk["conteudo"], metadata={"uf": chunk["uf"], "titulo": chunk["titulo"]})
    for chunk in dados
]

# -------------------------
# 🔎 EMBEDDINGS + BANCO VETORIAL
# -------------------------
embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#vector_db = FAISS.from_documents(documentos, embedding_engine)
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embedding_engine)

# -------------------------
# 🧐 MODELO E PROMPT
# -------------------------
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
prompt_com_fontes = PromptTemplate.from_template("""
Você é um assistente especializado em políticas públicas. Responda com base apenas no contexto abaixo.
Dos dados dos chunks veja o que melhor se adapta para a pergunat realizada. usar somente dados dos chunks.

Contexto:
{context}

Pergunta:
{question}
""")

n_documentos = 7

# -------------------------
# 🌏 MAPEAMENTO DE UFs
# -------------------------
ufs = {
    "acre": "AC", "alagoas": "AL", "amapa": "AP", "amazonas": "AM", "bahia": "BA",
    "ceara": "CE", "distrito federal": "DF", "espirito santo": "ES", "goias": "GO",
    "maranhao": "MA", "mato grosso": "MT", "mato grosso do sul": "MS", "minas gerais": "MG",
    "para": "PA", "paraiba": "PB", "parana": "PR", "pernambuco": "PE", "piaui": "PI",
    "rio de janeiro": "RJ", "rio grande do norte": "RN", "rio grande do sul": "RS",
    "rondonia": "RO", "roraima": "RR", "santa catarina": "SC", "sao paulo": "SP",
    "sergipe": "SE", "tocantins": "TO", "brasil": "NACIONAL"
}
ufs.update({sigla: sigla for sigla in ufs.values()})

# -------------------------
# 🔍 FUNÇÃO: EXTRAI UF DA PERGUNTA
# -------------------------
def extrair_uf(pergunta):
    pergunta = pergunta.lower()
    for nome, sigla in ufs.items():
        if nome in pergunta:
            return sigla
    return "NACIONAL"

# -------------------------
# 📄 FUNÇÃO: FORMATAR CONTEXTO
# -------------------------
def format_docs(documentos):
    return "\n\n".join(
        f"Título: {d.metadata['titulo']}\nUF: {d.metadata['uf']}\n{d.page_content}"
        for d in documentos
    )

# -------------------------
# 💬 FUNÇÃO PRINCIPAL: RESPONDER
# -------------------------
def responder(pergunta):
    uf_detectada = extrair_uf(pergunta)

    retriever = vector_db.as_retriever(
        search_kwargs={"k": n_documentos, "filter": {"uf": uf_detectada}}
    )

    # monta o pipeline corretamente
    rag_dinamico = (
        {
            "question": RunnablePassthrough(),     # passa a pergunta diretamente
            "context": retriever | format_docs      # monta o contexto com base na pergunta
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"[UF detectada: {uf_detectada}]")
    return rag_dinamico.invoke(pergunta)
