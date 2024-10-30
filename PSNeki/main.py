import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI

# Define a chave de API da OpenAI
os.environ["OPENAI_API_KEY"] = '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'

def splitar_documentos(docs):
    # Inicializa o divisor de texto
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # Divide os documentos
    splitted = splitter.split_documents(docs)
    return splitted

def criar_indice(textos):
    # Cria embeddings usando o modelo 'text-embedding-ada-002'
    embd = OpenAIEmbeddings()
    embd.model = "text-embedding-ada-002"
    # Cria um índice FAISS
    idx = FAISS.from_documents(textos, embd)
    return idx

def criar_chain(indice):
    # Inicializa o modelo de chatbot da OpenAI
    chatbot = ChatOpenAI()
    chatbot.model_name = "gpt-3.5-turbo" # Define o modelo a ser usado
    chatbot.temperature = 0  # Define a temperatura
    # Cria uma cadeia de QA
    chain = RetrievalQA.from_chain_type(llm=chatbot, chain_type="stuff", retriever=indice.as_retriever())
    return chain

if __name__ == "__main__":
    # Define o local onde os PDFs estão armazenados
    diretorio_pdfs = './arquivos'

    # Lista todos os arquivos no local especificado
    arquivos = os.listdir(diretorio_pdfs)
    docs = []
    # Para cada arquivo, carrega o conteúdo usando o PyPDFLoader
    for arquivo in arquivos:
        caminho = diretorio_pdfs + '/' + arquivo
        loader = PyPDFLoader(caminho)
        data = loader.load()
        # Adiciona cada documento carregado à lista de documentos
        for d in data:
            docs.append(d)

    textos_divididos = splitar_documentos(docs)
    indice = criar_indice(textos_divididos)
    qa_chain = criar_chain(indice)

    while True:
        pergunta = input("Faça uma pergunta (ou digite 'sair' para encerrar): ")
        if pergunta.lower() == 'sair':
            break
        resposta = qa_chain.invoke(pergunta)
        print("\nResposta:", resposta['result'])
        print("\n" + "="*60 + "\n")
