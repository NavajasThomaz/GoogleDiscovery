import os
import argparse
import requests
from tqdm import tqdm

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["ollama", "api"], required=True, help="Processamento local (ollama) ou online (api)")
parser.add_argument("--action", choices=["process", "query"], required=True, help="Processar embeddings ou realizar busca")
args = parser.parse_args()

folders = {
    "csv": "RAW/csv",
    "pdf": "RAW/pdf",
    "txt": "RAW/txt"
}

processed_file = "processed_files.txt"
if not os.path.exists(processed_file):
    open(processed_file, "w").close()

with open(processed_file, "r") as f:
    processed = set(line.strip() for line in f)

docs = []

if args.action == "process":
    for ext, folder in folders.items():
        files = [filename for filename in os.listdir(folder) if filename not in processed]
        print(f"Arquivos encontrados em {folder}: {files}")  # <-- Adicione esta linha
        for filename in tqdm(files, desc=f"Processando {ext.upper()}"):
            filepath = os.path.join(folder, filename)
            if ext == "csv":
                docs += CSVLoader(filepath).load()
            elif ext == "pdf":
                docs += PyPDFLoader(filepath).load()
            elif ext == "txt":
                docs += TextLoader(filepath).load()
            with open(processed_file, "a") as f:
                f.write(filename + "\n")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest") if args.mode == "ollama" else None # Adapte para API se necessário
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    print("Processamento de embeddings concluído.")

elif args.action == "query":
    # Carregue o vectorstore salvo anteriormente
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest") if args.mode == "ollama" else None
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    query = input("Digite sua pergunta: ")
    if not query:
        print("Necessário fornecer uma pergunta para busca.")
        exit(1)
    docs_encontrados = vectorstore.similarity_search(query)
    contexto = "\n".join([doc.page_content for doc in docs_encontrados])
    prompt = f"{contexto}\n\nPergunta: {query}\nResposta:"
    if args.mode == "ollama":
        llm = Ollama(model="qwen3:14b")
        resposta = llm(prompt)
        print(resposta)
    else:
        # Exemplo de chamada para uma API de LLM (substitua pela sua URL e payload)
        api_url = "https://api.seu-llm.com/v1/chat/completions"
        payload = {
            "model": "seu-modelo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512
        }
        headers = {"Authorization": "Bearer SEU_TOKEN_API"}
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            resposta = response.json()["choices"][0]["message"]["content"]
            print(resposta)
        else:
            print("Erro na chamada à API:", response.status_code, response.text)
