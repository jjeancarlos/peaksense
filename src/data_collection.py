import requests
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv  # <-- 1. IMPORTAR A BIBLIOTECA

# --- 1. Configuração de Caminhos ---
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path(os.getcwd()).resolve()

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Carregar o .env ---
print("Carregando variáveis de ambiente do arquivo .env...")
load_dotenv(BASE_DIR / ".env")  # <-- 2. CARREGAR O ARQUIVO .env

# --- 3. Configuração da API ---

# <-- 3. LER A CHAVE DO AMBIENTE (FORMA SEGURA)
API_KEY = os.getenv("API_KEY") 

if not API_KEY: # Se a chave não foi encontrada
    print("ERRO: A variável 'API_KEY' não foi encontrada.")
    print("Certifique-se de que ela existe no seu arquivo .env na raiz do projeto.")
    exit()

# Termos que queremos buscar
KEYWORDS = [
    "apagão", "blecaute", "consumo de energia", "ANEEL", "ONS",
    "onda de calor", "frio intenso", "subestação", "linha de transmissão"
]

QUERY = " OR ".join(KEYWORDS)

url = (
    f"https://newsapi.org/v2/everything?"
    f"q=({QUERY}) AND (Brasil OR Norte OR Nordeste OR Sul OR Sudeste)"
    f"&language=pt"
    f"&sortBy=publishedAt"
    f"&apiKey={API_KEY}" # <-- A chave é usada aqui
)

print(f"Buscando notícias com a query: {QUERY}")

# --- 4. Execução da Coleta ---

def fetch_and_save_news():
    try:
        response = requests.get(url)
        response.raise_for_status() 
        
        data = response.json()
        articles = data.get('articles', [])
        
        if not articles:
            print("Nenhum artigo encontrado para esta query.")
            return

        print(f"Sucesso! {len(articles)} artigos encontrados.")
        
        df_news = pd.DataFrame(articles)
        df_news['source_name'] = df_news['source'].apply(lambda x: x.get('name'))
        df_final = df_news[['publishedAt', 'source_name', 'title', 'description', 'content', 'url']]
        
        caminho_saida = DATA_RAW_DIR / "noticias_energia_raw.csv"
        df_final.to_csv(caminho_saida, index=False, sep='|', encoding='utf-8')
        
        print(f"Notícias salvas com sucesso em: {caminho_saida}")

    except requests.exceptions.HTTPError as http_err:
        print(f"Erro HTTP: {http_err}")
        print(f"Resposta da API: {response.text}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# --- Ponto de Entrada ---
if __name__ == "__main__":
    fetch_and_save_news()