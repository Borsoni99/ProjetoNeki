import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI

# Define a chave de API da OpenAI
os.environ["OPENAI_API_KEY"] = 'API_KEY'

def splitar_documentos(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted = splitter.split_documents(docs)
    return splitted

def criar_indice(textos):
    embd = OpenAIEmbeddings()
    embd.model = "text-embedding-ada-002"
    idx = FAISS.from_documents(textos, embd)
    return idx

def criar_chain(indice):
    chatbot = ChatOpenAI()
    chatbot.model_name = "gpt-3.5-turbo" 
    chatbot.temperature = 0  
    # Cria uma cadeia de QA(perguntas e respostas)
    chain = RetrievalQA.from_chain_type(llm=chatbot, chain_type="stuff", retriever=indice.as_retriever())
    return chain

if __name__ == "__main__":
    diretorio_pdfs = './arquivos'
    arquivos = os.listdir(diretorio_pdfs)
    docs = []
    
    for arquivo in arquivos:
        caminho = diretorio_pdfs + '/' + arquivo
        loader = PyPDFLoader(caminho)
        data = loader.load()
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
