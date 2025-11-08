import streamlit as st
import pandas as pd
import xgboost as xgb
from pathlib import Path
import matplotlib.pyplot as plt
import os

# --- 1. Configuração da Página e Caminhos ---
st.set_page_config(page_title="Otimizador Energético", layout="wide")

# Define o diretório base do projeto
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path(os.getcwd()).resolve()

DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# --- 2. Funções de Carregamento (com Cache) ---
@st.cache_resource
def load_model(regiao):
    """Carrega o modelo XGBoost treinado para uma região."""
    regiao_str = regiao.lower().replace('/', '_')
    caminho_modelo = MODELS_DIR / f"xgb_model_{regiao_str}.json"
    
    if not caminho_modelo.exists():
        st.error(f"Erro: Modelo não encontrado em {caminho_modelo}")
        st.error("Por favor, rode 'src/ml_pipeline.py' primeiro para treinar os modelos.")
        return None
        
    model = xgb.XGBRegressor()
    model.load_model(caminho_modelo)
    return model

@st.cache_data
def load_data():
    """Carrega o master dataset processado."""
    caminho_dados = DATA_PROCESSED_DIR / "master_dataset.csv"
    if not caminho_dados.exists():
        st.error(f"Erro: 'master_dataset.csv' não encontrado em {caminho_dados}")
        st.error("Por favor, rode 'src/data_processing.py' primeiro.")
        return None
        
    df = pd.read_csv(
        caminho_dados, 
        sep=';', 
        decimal=',',
        parse_dates=['ds']
    )
    df = df.rename(columns={'População 2024': 'Populacao'})
    return df

# --- 3. Interface do Dashboard ---

st.title("💡 Otimizador de Consumo Energético")
st.markdown("Dashboard de análise e previsão da demanda energética por região (Modelo V4: XGBoost-Only)")

# Carrega os dados
df_master = load_data()

if df_master is not None:
    
    # Lista de regiões + opção "Todas"
    regioes = ["Global (Todas as Regiões)"] + sorted(df_master['Região'].unique().tolist())
    
    # --- Barra Lateral (Sidebar) ---
    with st.sidebar:
        st.header("Configurações")
        regiao_selecionada = st.selectbox(
            "Selecione a Região:",
            options=regioes
        )

    st.header(f"Análise de Performance: {regiao_selecionada}")

    # Define as features que o modelo precisa
    features = [
        'Temperatura', 'Umidade', 'Populacao', 'e_feriado', 
        'dia_semana', 'dia_mes', 'semana_ano', 'mes', 'trimestre'
    ]

    # --- Lógica de Plotagem ---
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    if regiao_selecionada == "Global (Todas as Regiões)":
        df_plot = df_master.groupby('ds')['y'].sum().reset_index()
        df_plot = df_plot.rename(columns={'y': 'y_real'})
        df_plot['y_pred'] = None 
        st.info("Mostrando consumo real agregado para todas as regiões. A previsão é feita por região.")
        
        # Plota o Real Agregado
        ax.plot(df_plot['ds'], df_plot['y_real'], label='Consumo Real (Agregado)', color='blue')
    
    else:
        model = load_model(regiao_selecionada)
        
        if model:
            df_plot = df_master[df_master['Região'] == regiao_selecionada].copy()
            X_historical = df_plot[features]
            df_plot['y_pred'] = model.predict(X_historical)
            df_plot = df_plot.rename(columns={'y': 'y_real'})
            
            # Plota o Real
            ax.plot(df_plot['ds'], df_plot['y_real'], label='Valor Real', color='blue', alpha=0.8)
            # Plota a Previsão
            ax.plot(df_plot['ds'], df_plot['y_pred'], label='Previsão do Modelo (XGBoost)', color='red', linestyle='--')

    # --- Configurações comuns do Gráfico ---
    ax.set_title(f"Consumo de Energia - {regiao_selecionada}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Consumo (MWm)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("---")
    
    # ==================================================================
    # === AQUI ESTÁ A CORREÇÃO ===
    # ==================================================================
    
    st.subheader("Dados Brutos")
    
    # Se for 'Global', mostra o dataframe simples (sem features)
    if regiao_selecionada == "Global (Todas as Regiões)":
        st.dataframe(df_plot[['ds', 'y_real']].tail())
        
    # Se for uma região, mostra o dataframe completo (com features)
    else:
        st.dataframe(df_plot[['ds', 'y_real', 'y_pred'] + features].tail())