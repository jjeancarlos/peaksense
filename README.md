# 💡 Otimizador de Consumo Energético

[](https://www.python.org/downloads/)
[](https://streamlit.io)
[](https://xgboost.ai/)

Este é um projeto *full-stack* de Data Science para previsão de demanda energética no Brasil. O objetivo é construir um pipeline completo, desde a coleta de dados brutos (web-scraping) até um modelo de Machine Learning de alta precisão e um dashboard interativo.

O modelo principal **(V4: XGBoost-Only)** alcançou um **Erro Percentual Médio (MAPE) de 6.70%** na previsão de demanda diária, utilizando um pipeline de features otimizado.

<img width="745" height="905" alt="Image" src="https://github.com/user-attachments/assets/9d826b63-57a1-46ed-9362-6b43c9c09117" />

<img width="1790" height="690" alt="Image" src="https://github.com/user-attachments/assets/0e160a0f-7e3c-4d9e-ad62-504f59892f5e" />



## 🚀 Principais Funcionalidades

  * **Pipeline de ETL:** Scripts para coletar, processar, limpar e harmonizar dados de múltiplas fontes (INMET, CCEE, IBGE).
  * **Modelo Preditivo (ML):** Um modelo XGBoost (V4) treinado por região, capaz de prever a demanda energética (`MWm`) com base em clima, população e fatores sazonais.
  * **Pipeline de NLP:** Um coletor de notícias (via NewsAPI) com um pipeline de NLP (spaCy) para classificar eventos (ex: "Onda de Calor", "Apagão") e fornecer contexto qualitativo.
  * **Dashboard Interativo:** Uma aplicação web (`Streamlit`) que carrega os modelos V4 treinados e permite a análise visual das previsões.
  * **Análise Geoespacial Avançada:** Um notebook de análise (`Projeto_Energia_FINAL.ipynb`) que utiliza dados climáticos completos e `geopandas` para explorar a distribuição espacial do clima e seu impacto no consumo.

-----

## 📊 Dashboard em Ação

O dashboard `src/dashboard.py` é o produto final do pipeline principal, onde os modelos V4 são carregados para um *backtest* visual.

**(Cole um GIF ou screenshot do seu `streamlit run src/dashboard.py` aqui\!)**
`![Demo do Dashboard](caminho/para/seu/screenshot.png)`

-----

## 🛠️ Stack Tecnológico

Este projeto utiliza um conjunto de ferramentas modernas de Data Science, conforme definido no `requirements.txt`:

  * **Núcleo de Dados e ML:** `pandas`, `numpy`, `scikit-learn`, `xgboost`
  * **Séries Temporais:** `prophet`, `cmdstanpy`, `holidays`
  * **Geoprocessamento (Mapas):** `geopandas`, `geobr`
  * **Processamento de Linguagem Natural (PLN / NLP):** `spacy`
  * **Exploração, Visualização e Notebooks:** `jupyter`, `matplotlib`, `seaborn`, `chardet`
  * **Dashboard / Aplicação Web:** `streamlit`
  * **Coleta de Dados e APIs:** `requests`
  * **Gerenciamento de Variáveis de Ambiente:** `python-dotenv`

-----

## 📂 Estrutura do Projeto

O projeto segue uma estrutura de Data Science padrão, separando dados, notebooks e scripts de produção.

```bash
peaksense/
├── .env                  # [SECRETO] Armazena a API Key 
├── .gitignore            # Ignora arquivos de ambiente, dados e modelos
├── README.md             # Este arquivo
├── requirements.txt      # Lista de dependências do projeto
│
├── data/
│   ├── raw/              # Dados brutos (CSVs originais, notícias_raw.csv, dados INMET completos)
│   ├── processed/        # Dados limpos para o pipeline principal (master_dataset.csv)
│   └── models/           # Modelos XGBoost (V4) treinados e salvos (.json)
│
├── notebooks/            # Notebooks Jupyter para exploração e avaliação
│   ├── eda.ipynb         # (Caminho A) Análise Exploratória (Objetivos A, B, C)
│   ├── model_evaluation.ipynb # (Caminho A) A jornada de V1 a V4 (Provas de Modelo)
│   ├── nlp_analysis.ipynb     # (Caminho A) Pipeline de NLP com spaCy
│   └── Projeto_Energia_FINAL.ipynb # (Caminho B) Análise avançada com Geopandas e dados completos
│
└── src/
    ├── __init__.py
    ├── data_collection.py     # (Caminho A) Coleta notícias da NewsAPI
    ├── data_processing.py     # (Caminho A) Limpa e junta os 3 CSVs -> master_dataset.csv
    ├── ml_pipeline.py         # (Caminho A) Treina o modelo V4 e salva em /data/models
    └── dashboard.py           # (Caminho A) Roda o dashboard Streamlit
```

-----

## ⚙️ Instalação e Execução

Siga estes passos para configurar e rodar o projeto localmente.

### 1\. Pré-requisitos

  * Python 3.10 ou superior
  * Chave de API gratuita da [NewsAPI.org](https://newsapi.org/) (necessária para o pipeline de NLP)

### 2\. Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/peaksense.git
cd peaksense

# 2. Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
# .\venv\Scripts\activate  # (Windows)

# 3. Instale todas as dependências do Python
# (Isso pode levar algum tempo devido ao geopandas e prophet)
pip install -r requirements.txt

# 4. Baixe o modelo de linguagem treinado para o português (do spaCy)
python -m spacy download pt_core_news_lg
```

### 3\. Configuração de Segurança (API Key)

Este é um passo **crucial** para proteger sua chave de API (usada pelo `data_collection.py`).

1.  Crie um arquivo chamado `.env` na raiz do projeto (`peaksense/.env`).
2.  Abra o arquivo e cole sua chave de API da NewsAPI:
    ```bash
    API_KEY="sua_chave_real_da_newsapi_aqui"
    ```
3.  O arquivo `.gitignore` já está configurado para **nunca** enviar seu `.env` para o GitHub.

-----

## 🚀 Modo de Uso (Dois Caminhos)

Este projeto oferece dois caminhos de análise:

### Caminho A: Pipeline de Produção (Modelo V4 + Dashboard)

Execute os scripts na ordem correta para popular os dados, treinar o modelo e iniciar o dashboard.

**Certifique-se de que seu ambiente virtual (`venv`) está ativado para todos os passos.**

```bash
# PASSO 1: Processar Dados de ML (Obrigatório)
# Lê os 3 CSVs básicos de /raw, limpa, junta e salva em data/processed/master_dataset.csv
python src/data_processing.py

# PASSO 2: Treinar o Modelo Final (Obrigatório)
# Carrega o master_dataset.csv, treina os 4 modelos (V4) e salva em /data/models/
python src/ml_pipeline.py

# PASSO 3: Iniciar o Dashboard (O Produto Final)
# Inicia a aplicação web localmente
streamlit run src/dashboard.py
```

Acesse `http://localhost:8501` no seu navegador para ver o dashboard.

### Caminho B: Análise Geoespacial Avançada (Notebook Bônus)

Este caminho usa um conjunto de dados climáticos mais completo e não está conectado ao pipeline principal do dashboard.

1.  **Baixe os Dados:** Faça o download dos dados climáticos completos do INMET [neste link do Google Drive](https://drive.google.com/drive/folders/19UBBJoI2rACpZB1SK68ZeWd5aAH37Nzg?usp=sharing).
2.  **Organize os Arquivos:** Coloque os arquivos baixados na pasta `data/raw/` (ou atualize os caminhos dentro do notebook).
3.  **Execute o Notebook:** Abra e execute as células do `notebooks/Projeto_Energia_FINAL.ipynb` usando o Jupyter.

-----

## 🌍 Fontes dos Dados

Os dados brutos para este projeto foram obtidos de fontes públicas oficiais brasileiras:

  * **Dados Climáticos (Pipeline Principal):** [INMET - Instituto Nacional de Meteorologia](https://bdmep.inmet.gov.br/)
  * **Dados Climáticos (Análise Geoespacial):** [Dataset INMET Compilado (via Google Drive)](https://drive.google.com/drive/folders/19UBBJoI2rACpZB1SK68ZeWd5aAH37Nzg?usp=sharing)
  * **Consumo de Energia:** [CCEE - Câmara de Comercialização de Energia Elétrica](https://www.ccee.org.br/)
  * **Dados Populacionais:** [IBGE - Instituto Brasileiro de Geografia e Estatística](https://www.ibge.gov.br/)
  * **Dados de Notícias (NLP):** [NewsAPI.org](https://newsapi.org/)

-----

## 🔬 Metodologia e Descobertas Chave

A análise completa está nos notebooks (`/notebooks`), mas os principais insights são:

### 1\. Análise Exploratória (Objetivos A e B)

  * **Consumo × População (A):** Hipótese **confirmada**. A correlação entre População e Consumo total é de **0.96**.
  * **Consumo × Clima (B):** Hipótese **confirmada (com nuances)**. O clima tem um impacto *não-linear* e *regional*:
      * **Região Sul:** Mais frio = Mais consumo (efeito de aquecedores).
      * **Região Sudeste/CO:** Mais calor = Mais consumo (efeito de ar-condicionado).

### 2\. A Jornada do Modelo (V1 → V4)

O modelo final do pipeline (V4) foi escolhido após um processo rigoroso de avaliação (ver `model_evaluation.ipynb`):

  * **V1 (Prophet + Clima):** `FALHOU` (MAPE \> 1000%). O Prophet **extrapolou** os regressores de clima.
  * **V2/V3 (Híbrido Prophet + XGBoost):** `FALHOU`. Arquitetura instável que previu consumo *negativo*.
  * **V4 (XGBoost-Only):** `SUCESSO!` (MAPE 6.70%). Um único modelo XGBoost (treinado por região) provou ser robusto e aprendeu as regras não-lineares do clima.

### 3\. Resultados Finais (V4: XGBoost-Only)

| Região | Erro Médio (MAPE) | Erro Médio (MAE) |
| :--- | :---: | :---: |
| Nordeste | 5.04 % | 410.73 MWm |
| Norte | 5.54 % | 219.38 MWm |
| Sudeste/CO | 7.38 % | 1649.37 MWm |
| Sul | 8.85 % | 605.41 MWm |
| **GLOBAL (Agregado)** | **6.70 %** | **721.22 MWm** |

### 4\. Pipeline de NLP (Contexto Qualitativo)

O pipeline de NLP (`nlp_analysis.ipynb`) provou ser eficaz em filtrar "ruído" (94% das notícias) e identificar eventos reais, como **"alertas de tempestade do Inmet"**, fornecendo um contexto valioso para explicar picos de erro no modelo de ML.

### 5\. Análise Bônus: Geoprocessamento (`Projeto_Energia_FINAL.ipynb`)

Este notebook paralelo utiliza os dados climáticos completos do INMET e as bibliotecas `geopandas`/`geobr` para criar uma análise geoespacial. Ele explora como as variáveis climáticas (temperatura, precipitação) se distribuem *espacialmente* pelo Brasil e como isso se correlaciona com os centros de consumo regionais, enriquecendo a análise exploratória.

## 👤 Autores
  * [Jean Carlos](https://github.com/jjeancarlos)
  * [Matheus Menezes](https://github.com/MatheusLuv)
  * [Tiago Elias](https://github.com/TiagosailE)

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
