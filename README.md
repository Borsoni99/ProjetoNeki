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


## Passo a passo para utilização

### Preparação do ambiente:

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

## Exemplos de interações com o sistema e resultados obtidos.

#### Pergunta:
Qual é o tema principal do documento que fala sobre feedback de produto.

#### Resposta: 
O tema principal do documento que fala sobre feedback de produto é a funcionalidade de Captura de Feedbacks sobre o Produto na plataforma IQuirium. Essa funcionalidade permite que os usuários forneçam feedback diretamente na plataforma sobre sua experiência de uso, funcionalidades, desempenho e sugestões de melhoria. O documento aborda como os feedbacks serão armazenados, analisados pela equipe da plataforma, e como serão utilizados para identificar áreas de melhoria e novas funcionalidades a serem desenvolvidas.

![image](https://github.com/user-attachments/assets/47f103e5-e9f0-42dd-82e9-919834e4f331)

#### Pergunta:
A jornada do usuário no PDF de feedback de produtos é dividida em duas partes:

#### Resposta: 
A jornada do usuário no PDF de feedback de produtos é dividida em duas partes:

1. Envio de Feedback Espontâneo:
- O usuário acessa a seção Feedback no menu da plataforma.
- A plataforma exibe um formulário simples para o usuário relatar sua experiência, incluindo campos opcionais para:
  - Comentários gerais.
  - Sugestões de melhoria.
  - Relato de problemas técnicos ou erros.
- O usuário preenche os campos desejados e envia o feedback.
- A plataforma confirma o recebimento do feedback e exibe uma mensagem de agradecimento.

2. Solicitar Feedback de Outro Usuário:
- O usuário acessa a seção Feedback entre Usuários e escolhe a opção "Solicitar Feedback".
- O usuário seleciona o destinatário (um colega ou mentor da plataforma) e define o tipo de feedback que deseja receber (ex.: feedback sobre competências, comportamentos, ou uma atividade específica).

Essas são as etapas que os usuários seguem ao interagir com a funcionalidade de feedback de produtos no PDF.

![image](https://github.com/user-attachments/assets/8632d29c-3d78-4950-b223-9556a845f8e9)


