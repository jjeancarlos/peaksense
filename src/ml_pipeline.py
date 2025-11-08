import pandas as pd
import xgboost as xgb
from pathlib import Path
import os
import warnings

# Ignorar avisos
warnings.filterwarnings('ignore')

# --- 1. Configuração de Caminhos ---
# Define o diretório base do projeto (a pasta 'projeto/')
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path(os.getcwd()).resolve()

DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Garante que a pasta 'models' exista
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Diretório Base: {BASE_DIR}")
print(f"Pasta de Dados Processados: {DATA_PROCESSED_DIR}")
print(f"Pasta de Modelos: {MODELS_DIR}")

# --- 2. Função de Treinamento ---

def load_data():
    """Carrega o master dataset processado."""
    print("Carregando master_dataset.csv...")
    try:
        df = pd.read_csv(
            DATA_PROCESSED_DIR / "master_dataset.csv", 
            sep=';', 
            decimal=',',
            parse_dates=['ds']
        )
        # Renomear colunas se necessário (para o XGBoost)
        df = df.rename(columns={'População 2024': 'Populacao'})
        print("✅ Dados carregados.")
        return df
    except FileNotFoundError:
        print(f"❌ ERRO: 'master_dataset.csv' não encontrado em {DATA_PROCESSED_DIR}")
        print("Por favor, rode 'src/data_processing.py' primeiro.")
        return None

def train_and_save_models(df):
    """Treina o modelo V4 (XGBoost-Only) em 100% dos dados e salva."""
    if df is None:
        print("Treinamento cancelado devido a erro no carregamento.")
        return

    # Esta é a lista de features vencedora da nossa V4
    features = [
        'Temperatura', 'Umidade', 'Populacao', 'e_feriado', 
        'dia_semana', 'dia_mes', 'semana_ano', 'mes', 'trimestre'
    ]
    
    # O alvo (target) que queremos prever
    target = 'y'

    print("Iniciando treinamento dos 4 modelos regionais...")

    # Loop por cada região
    for regiao in df['Região'].unique():
        print(f"\n--- Treinando para Região: {regiao} ---")
        
        # Filtra os dados apenas para esta região
        df_regional = df[df['Região'] == regiao].copy()
        
        X_train = df_regional[features]
        y_train = df_regional[target]
        
        # Define o modelo XGBoost
        # (Usamos os mesmos parâmetros do notebook)
        model_xgb = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            random_state=42
            # NOTA: Removemos 'early_stopping_rounds' 
            # pois estamos treinando no set COMPLETO (não há set de validação)
        )
        
        # Treina o modelo em 100% dos dados da região
        model_xgb.fit(X_train, y_train)
        
        # Define o caminho de saída
        caminho_modelo = MODELS_DIR / f"xgb_model_{regiao.lower().replace('/', '_')}.json"
        
        # Salva o modelo
        model_xgb.save_model(caminho_modelo)
        
        print(f"  ✅ Modelo para {regiao} treinado e salvo em:")
        print(f"     {caminho_modelo}")

# --- 3. Função Principal ---

def main():
    """Orquestra o pipeline de ML: carrega dados, treina e salva modelos."""
    print("--- INICIANDO PIPELINE DE TREINAMENTO DE ML ---")
    df = load_data()
    train_and_save_models(df)
    print("\n--- PIPELINE DE TREINAMENTO CONCLUÍDO ---")

# --- Ponto de Entrada ---
if __name__ == "__main__":
    main()