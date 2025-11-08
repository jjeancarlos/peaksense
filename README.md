# 💡 Otimizador de Consumo Energético

[](https://www.python.org/downloads/)
[](https://streamlit.io)
[](https://xgboost.ai/)

Este é um projeto *full-stack* de Data Science para previsão de demanda energética no Brasil. O objetivo é construir um pipeline completo, desde a coleta de dados brutos (web-scraping) até um modelo de Machine Learning de alta precisão e um dashboard interativo para consumo dos resultados.

O modelo final **(V4: XGBoost-Only)** alcançou um **Erro Percentual Médio (MAPE) de 6.70%** na previsão de demanda diária, com performance robusta em todas as regiões do país.

-----

## 🚀 Principais Funcionalidades

  * **Pipeline de ETL:** Scripts para coletar, processar, limpar e harmonizar dados de múltiplas fontes (INMET, CCEE, IBGE).
  * **Modelo Preditivo (ML):** Um modelo XGBoost treinado por região, capaz de prever a demanda energética (`MWm`) com base em clima, população e fatores sazonais.
  * **Pipeline de NLP:** Um coletor de notícias (via NewsAPI) com um pipeline de NLP (spaCy) para classificar eventos (ex: "Onda de Calor", "Apagão") e fornecer contexto qualitativo para anomalias no consumo.
  * **Dashboard Interativo:** Uma aplicação web (`Streamlit`) que carrega os modelos treinados e permite a análise visual das previsões contra os dados reais.

-----

## 📊 Dashboard em Ação

O dashboard `src/dashboard.py` é o produto final do projeto, onde os modelos treinados são carregados e usados para um *backtest* visual.

## 🛠️ Stack Tecnológico

Este projeto utiliza um conjunto de ferramentas modernas de Data Science:

  * **Manipulação de Dados:** `pandas`, `numpy`
  * **Machine Learning:** `scikit-learn` (Métricas), `xgboost` (Modelo V4), `prophet` (Testes V1-V3)
  * **Coleta de Dados & NLP:** `requests` (API), `python-dotenv` (Segurança), `spacy` (pt\_core\_news\_lg)
  * **Engenharia de Features:** `holidays` (Feriados)
  * **Visualização & Dashboard:** `matplotlib`, `seaborn`, `streamlit`
  * **Ambiente & Notebooks:** `venv`, `jupyter`, `notebook`

-----

## 📂 Estrutura do Projeto

O projeto segue uma estrutura de Data Science padrão, separando dados brutos, processados, notebooks de exploração e scripts de produção.

```bash
peaksense/
├── .env                  # [SECRETO] Armazena a API Key (ignorado pelo Git)
├── .gitignore            # Ignora arquivos de ambiente, dados e modelos
├── README.md             # Este arquivo
├── requirements.txt      # Lista de dependências do projeto
│
├── data/
│   ├── raw/              # Dados brutos (CSVs originais, notícias_raw.csv)
│   └── processed/        # Dados limpos e prontos para ML (master_dataset.csv)
│
├── models/               # Modelos XGBoost treinados e salvos (.json)
│
├── notebooks/            # Notebooks Jupyter para exploração e avaliação
│   ├── eda.ipynb         # Análise Exploratória (Objetivos A, B, C)
│   ├── model_evaluation.ipynb # A jornada de V1 a V4 (Provas de Modelo)
│   └── nlp_analysis.ipynb     # Pipeline de NLP com spaCy
│
└── src/
    ├── __init__.py
    ├── data_collection.py     # (NLP) Coleta notícias da NewsAPI
    ├── data_processing.py     # (ETL) Limpa e junta os 3 CSVs -> master_dataset.csv
    ├── ml_pipeline.py         # (ML) Treina o modelo V4 e salva em /models
    └── dashboard.py           # (App) Roda o dashboard Streamlit
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
.\venv\Scripts\activate  # (Windows)

# 3.modelo de linguagem treinado para o português
python -m spacy download pt_core_news_lg

# 4. Instale todas as dependências
pip install -r requirements.txt
```

### 3\. Configuração de Segurança (API Key)

Este é um passo **crucial** para proteger sua chave de API.

1.  Crie um arquivo chamado `.env` na raiz do projeto (`peaksense/.env`).
2.  Abra o arquivo e cole sua chave de API da NewsAPI:

    ```bash
    API_KEY="sua_chave_real_da_newsapi_aqui"
    ```
3.  O arquivo `.gitignore` já está configurado para **nunca** enviar seu `.env` para o GitHub.

-----

## 🚀 Modo de Uso (Executando o Pipeline)

Execute os scripts na ordem correta para popular os dados, treinar o modelo e iniciar o dashboard.

**Certifique-se de que seu ambiente virtual (`venv`) está ativado para todos os passos.**

### Passo 1: Coletar Dados de NLP

(Opcional, mas necessário para o `nlp_analysis.ipynb`)

```bash
# Busca notícias recentes e salva em data/raw/noticias_energia_raw.csv
python src/data_collection.py
```

### Passo 2: Processar Dados de ML

(Obrigatório)

```bash
# Lê os 3 CSVs de /raw, limpa, junta e salva em data/processed/master_dataset.csv
python src/data_processing.py
```

### Passo 3: Treinar o Modelo Final

(Obrigatório)

```bash
# Carrega o master_dataset.csv, treina os 4 modelos (V4) e salva em /models/
python src/ml_pipeline.py
```

### Passo 4: Iniciar o Dashboard

(O Produto Final)

```bash
# Inicia a aplicação web localmente
streamlit run src/dashboard.py
```

Acesse `http://localhost:8501` no seu navegador para ver o dashboard.

-----

## 🌍 Fontes dos Dados

Os dados brutos para este projeto foram obtidos de fontes públicas oficiais brasileiras:

  * **Dados Climáticos:** [INMET - Instituto Nacional de Meteorologia](https://bdmep.inmet.gov.br/)
  * **Consumo de Energia:** [CCEE - Câmara de Comercialização de Energia Elétrica](https://www.ccee.org.br/)
  * **Dados Populacionais:** [IBGE - Instituto Brasileiro de Geografia e Estatística](https://www.ibge.gov.br/)
  * **Dados de Notícias (NLP):** [NewsAPI.org](https://newsapi.org/)

-----

## 🔬 Metodologia e Descobertas Chave

A análise completa está nos notebooks (`/notebooks`), mas os principais insights são:

### 1\. Análise Exploratória (Objetivos A e B)

  * **Consumo × População (A):** Hipótese **confirmada**. A correlação entre População e Consumo total é de **0.96**, provando ser o driver macro mais importante.
  * **Consumo × Clima (B):** Hipótese **confirmada (com nuances)**. O clima tem um impacto *não-linear* e *regional*:
      * **Região Sul:** Mais frio = Mais consumo (efeito de aquecedores).
      * **Região Sudeste/CO:** Mais calor = Mais consumo (efeito de ar-condicionado).

### 2\. A Jornada do Modelo (V1 → V4)

O modelo final (V4) foi escolhido após um processo rigoroso de avaliação (ver `model_evaluation.ipynb`):

  * **V1 (Prophet + Clima):** `FALHOU` (MAPE \> 1000%). O Prophet **extrapolou** os regressores de clima, prevendo valores absurdos.
  * **V2/V3 (Híbrido Prophet + XGBoost nos Resíduos):** `FALHOU`. Arquitetura instável que previu consumo *negativo* e padrões invertidos.
  * **V4 (XGBoost-Only):** `SUCESSO!` (MAPE 6.70%). Um único modelo XGBoost (treinado por região) provou ser robusto, aprendeu as regras não-lineares do clima e não sofreu de extrapolação.

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

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.