
import streamlit as st

chiave = "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"
# chiave = st.secrets["superkey"]

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# from PIL import Image
# logo = Image.open("Chatbot.webp")
# st.image(logo)

st.header("Il mio chatbot")

with st.sidebar:
    st.title("I miei documenti")
    file = st.file_uploader("Carica un file PDF e chiedi le info che ti servono", type="pdf")

from PyPDF2 import PdfReader

if file is not None:
    testo_letto = PdfReader(file)

    testo = ""
    for pagina in testo_letto.pages:
        testo = testo + pagina.extract_text()
        # st.write(testo)

    # Usiamo il text splitter di Langchain
    testo_spezzato = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000, # Numero di caratteri per chunk
        chunk_overlap=150,
        length_function=len
        )

    pezzi = testo_spezzato.split_text(testo)
    # st.write(pezzi)

    # Generazione embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=chiave)

    # Vector store - FAISS (by Facebook)
    vector_store = FAISS.from_texts(pezzi, embeddings)

    # Prompt
    domanda = st.text_input("Chiedi al chatbot:")

    if domanda:
        rilevanti = vector_store.similarity_search(domanda)
        # st.write(match)

        # Definiamo l'LLM
        llm = ChatOpenAI(
            openai_api_key = chiave,
            temperature = 1.0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo-0125")
        # https://platform.openai.com/docs/models/compare

        # Output
        # Chain: prindi la domanda, individua i frammenti rilevanti,
        #  passali all'LLM, genera la risposta
        chain = load_qa_chain(llm, chain_type="stuff")
        risposta = chain.run(input_documents = rilevanti, question = domanda)
        st.write(risposta)
