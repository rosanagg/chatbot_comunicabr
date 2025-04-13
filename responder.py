from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Carrega as variáveis do .env
#load_dotenv()
OPENAI_API_KEY = "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"

# Recarrega o banco vetorial salvo
embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = Chroma.from_documents(documentos, embedding_engine)
#vector_db = Chroma(persist_directory="chroma_db", embedding_function=embedding_engine)


# Puxa o prompt do LangChain Hub
prompt = hub.pull("rlm/rag-prompt")

# Inicializa o modelo da OpenAI
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Quantidade de documentos a recuperar por pergunta
n_documentos = 5

# Dicionário de UFs
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

# Extrai a UF da pergunta
def extrair_uf(pergunta):
    pergunta = pergunta.lower()
    for nome, sigla in ufs.items():
        if nome in pergunta:
            return sigla
    return "NACIONAL"

# Formata os documentos para o prompt
def format_docs(documentos):
    return "\n\n".join(documento.page_content for documento in documentos)

# Função principal que será usada no Streamlit
def responder(pergunta):
    uf_detectada = extrair_uf(pergunta)

    retriever = vector_db.as_retriever(
        search_kwargs={"k": n_documentos, "filter": {"uf": uf_detectada}}
    )

    rag_dinamico = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | format_docs
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    resposta = rag_dinamico.invoke(pergunta)
    return f"🗺️ UF detectada: **{uf_detectada}**\n\n{resposta}"
