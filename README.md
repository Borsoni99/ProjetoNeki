# ProjetoNeki

## Documentação Detalhada

### objetivo do código
Criar um chatbot que responda perguntas do usuário com base no conteúdo de arquivos pdf. para isso, o código:

- carrega documentos pdf de um diretório específico.
- divide os documentos em pedaços menores (*chunks*) para facilitar o processamento e a indexação.
- cria embeddings dos textos usando um modelo de linguagem da openai para permitir a busca eficiente.
- configura um índice de busca usando a biblioteca *faiss* para facilitar a recuperação dos documentos relevantes.
- define uma cadeia de perguntas e respostas que utiliza os embeddings para gerar respostas em linguagem natural com base nos documentos carregados.

### estrutura do código

o código é composto por 3 partes principais:

1. **importação das bibliotecas necessárias**
2. **definição de funções auxiliares**
3. **execução principal do programa**

---

### 1. importação das bibliotecas necessárias:

```python
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
```
- **os**: interage com o sistema operacional.
- **PyPDFLoader**: biblioteca para carregar documentos pdf.
- **RecursiveCharacterTextSplitter**: divide textos em pedaços menores
- **OpenAIEmbeddings**: gera embeddings dos textos.
- **FAISS**: biblioteca para indexação e busca eficiente de vetores.
- **RetrievalQA**: cria uma QA baseada nos documentos fornecidos.
- **ChatOpenAI**: modelo de linguagem natural da openai.

### 2. definição de funções auxiliares

#### função `splitar_documentos(docs)`

```python
def splitar_documentos(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted = splitter.split_documents(docs)
    return splitted
```
objetivo: dividir documentos longos em pedaços menores para facilitar o processamento.

#### função `criar_indice(texts)`

```python
def criar_indice(texts):
    embd = OpenAIEmbeddings()
    embd.model = "text-embedding-ada-002"
    idx = FAISS.from_documents(texts, embd)
    return idx
```
objetivo: cria um objeto com os embeddings dos textos.

#### Função criar_chain(indice)

```python
def criar_chain(indice):
    chatbot = ChatOpenAI()
    chatbot.model_name = "gpt-3.5-turbo"
    chatbot.temperature = 0
    chain = RetrievalQA.from_chain_type(
        llm=chatbot,
        chain_type="stuff",
        retriever=indice.as_retriever()
    )
    return chain
```
Objetivo: Criar um objeto chain que será usado para gerar perguntas e respostas para o usuário


### 3. execução principal do programa

- **carrega os documentos**
```python
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
```
- **chama as funções auxiliares**
```python
    textos_divididos = splitar_documentos(docs)
    indice = criar_indice(textos_divididos)
    qa_chain = criar_chain(indice)
```
- **interage com o usuário**
```python
    while True:
        pergunta = input("Faça uma pergunta (ou digite 'sair' para encerrar): ")
        if pergunta.lower() == 'sair':
            break
        resposta = qa_chain.invoke(pergunta)
        print("\nResposta:", resposta['result'])
        print("\n" + "="*60 + "\n")
```


## passo a passo para utilização

### preparação do ambiente:

1. certifique-se de ter o python instalado em sua máquina.
2. instale as bibliotecas necessárias:

   ```bash
   pip install langchain openai faiss-cpu pypdf
   ```

1. crie uma pasta chamada `arquivos` no mesmo diretório do seu script python.
2. coloque todos os arquivos pdf que deseja utilizar dentro dessa pasta.

### configuração da chave de api:

- no código, substitua `'API_KEY'` pela sua chave de api real da openai.

### execução do código:

1. execute o script python:

   ```bash
   python seu_script.py
   ```
- o programa solicitará que você faça uma pergunta.

### interação:

1. digite uma pergunta relacionada ao conteúdo dos pdfs fornecidos.
2. o programa processará a pergunta e exibirá a resposta.
3. para encerrar o programa, digite `'sair'`.
