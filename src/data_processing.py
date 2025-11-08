import pandas as pd
import numpy as np
import holidays  # Biblioteca para pegar feriados
from pathlib import Path
import os
import warnings

# Ignorar avisos que podem aparecer durante a limpeza
warnings.filterwarnings('ignore')

# --- 1. Configuração de Caminhos ---
# Define o diretório base do projeto (a pasta 'projeto/')
# __file__ se refere a este arquivo (data_processing.py)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Garante que a pasta 'processed' exista
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print(f"Diretório Base: {BASE_DIR}")
print(f"Pasta Raw: {DATA_RAW_DIR}")
print(f"Pasta Processed: {DATA_PROCESSED_DIR}")


# --- 2. Funções de Carga e Limpeza ---

def load_raw_data():
    """Carrega os 3 arquivos CSV da pasta raw."""
    print("Carregando dados brutos...")
    try:
        df_consumo = pd.read_csv(
            DATA_RAW_DIR / "consumo_historico_por_regiao.csv",
            sep='\t', encoding='utf-16', decimal=','
        )
        
        df_pop = pd.read_csv(
            DATA_RAW_DIR / "crescimento_populacional_regioes_2020_2024.csv",
            sep=',', encoding='utf-8', decimal=','
        )
        # Remove a linha "Total/Brasil" se houver
        df_pop = df_pop[~df_pop['Região'].str.contains('Brasil|Total', case=False, na=False)]

        
        df_clima = pd.read_csv(
            DATA_RAW_DIR / "medias_temperatura_umidade_2024.csv",
            sep=',', encoding='utf-8', decimal=','
        )
        print("✅ Dados brutos carregados.")
        return df_consumo, df_pop, df_clima
        
    except FileNotFoundError as e:
        print(f"❌ ERRO: Arquivo não encontrado. {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ ERRO ao carregar dados: {e}")
        return None, None, None

def clean_data(df_consumo, df_pop, df_clima):
    """Limpa e formata os DataFrames."""
    print("Iniciando limpeza...")
    
    # --- Limpando Consumo ---
    try:
        df_consumo['Consumo_Limpo'] = df_consumo['Consumo (MWm)'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df_consumo['Consumo_Limpo'] = pd.to_numeric(df_consumo['Consumo_Limpo'], errors='coerce')
        df_consumo['Data'] = pd.to_datetime(df_consumo['Data'], format='%d/%m/%Y')
        df_consumo['Ano'] = df_consumo['Data'].dt.year
        df_consumo['Mes_Num'] = df_consumo['Data'].dt.month
        df_consumo = df_consumo.rename(columns={'Submercado': 'Região'})
        df_consumo['Região'] = df_consumo['Região'].str.strip()
        
        # Agrega por dia e região
        df_consumo_agg = df_consumo.groupby(['Data', 'Região', 'Ano', 'Mes_Num'])['Consumo_Limpo'].sum().reset_index()
        print("✅ df_consumo limpo e agregado por dia.")
    except Exception as e:
        print(f"❌ ERRO ao limpar df_consumo: {e}")
        return None, None, None

    # --- Limpando Clima ---
    try:
        df_clima['Temperatura'] = pd.to_numeric(df_clima['Temperatura (°C)'].str.replace(',', '.', regex=False), errors='coerce')
        df_clima['Umidade'] = pd.to_numeric(df_clima['Umidade (%)'].str.replace(',', '.', regex=False), errors='coerce')
        mapa_meses = {'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4, 'Maio': 5, 'Junho': 6, 'Julho': 7, 'Agosto': 8, 'Setembro': 9, 'Outubro': 10, 'Novembro': 11, 'Dezembro': 12}
        df_clima['Mes_Num'] = df_clima['Mês'].map(mapa_meses)
        df_clima['Ano'] = 2024 # Dado é de 2024
        df_clima['Região'] = df_clima['Região'].str.strip()
        print("✅ df_clima limpo.")
    except Exception as e:
        print(f"❌ ERRO ao limpar df_clima: {e}")
        return None, None, None

    # --- Limpando População ---
    try:
        df_pop['Crescimento_Float'] = pd.to_numeric(df_pop['Crescimento (%)'].str.replace(',', '.', regex=False), errors='coerce')
        df_pop['Região'] = df_pop['Região'].str.strip()
        df_pop['Ano'] = 2024 # Dado é de 2024
        print("✅ df_pop limpo.")
    except Exception as e:
        print(f"❌ ERRO ao limpar df_pop: {e}")
        return None, None, None
        
    return df_consumo_agg, df_pop, df_clima

def harmonize_regions(df_pop, df_clima):
    """Harmoniza as regiões de População e Clima para bater com Consumo."""
    print("Harmonizando regiões...")
    
    # --- Harmonizando df_pop ---
    try:
        pop_se = df_pop[df_pop['Região'] == 'Sudeste']['População 2024'].iloc[0]
        pop_co = df_pop[df_pop['Região'] == 'Centro-Oeste']['População 2024'].iloc[0]
        
        nova_linha_pop = {'Região': 'Sudeste/Centro-Oeste', 'População 2024': pop_se + pop_co, 'Ano': 2024}
        df_pop_harmonizado = df_pop[~df_pop['Região'].isin(['Sudeste', 'Centro-Oeste'])].copy()
        df_pop_harmonizado = pd.concat([df_pop_harmonizado, pd.DataFrame([nova_linha_pop])], ignore_index=True)
        print("✅ df_pop harmonizado.")
    except Exception as e:
        print(f"❌ ERRO ao harmonizar df_pop: {e}")
        return None, None

    # --- Harmonizando df_clima (Média Ponderada) ---
    try:
        pop_lookup = df_pop.set_index('Região')['População 2024']
        pop_se_2024 = pop_lookup['Sudeste']
        pop_co_2024 = pop_lookup['Centro-Oeste']
        pop_total_seco = pop_se_2024 + pop_co_2024
        
        clima_se = df_clima[df_clima['Região'] == 'Sudeste'].set_index('Mes_Num')
        clima_co = df_clima[df_clima['Região'] == 'Centro-Oeste'].set_index('Mes_Num')
        
        temp_ponderada = (clima_se['Temperatura'] * pop_se_2024 + clima_co['Temperatura'] * pop_co_2024) / pop_total_seco
        umidade_ponderada = (clima_se['Umidade'] * pop_se_2024 + clima_co['Umidade'] * pop_co_2024) / pop_total_seco
        
        df_clima_seco = pd.DataFrame({
            'Região': 'Sudeste/Centro-Oeste', 'Mes_Num': temp_ponderada.index,
            'Temperatura': temp_ponderada.values, 'Umidade': umidade_ponderada.values, 'Ano': 2024
        })
        
        df_clima_harmonizado = df_clima[~df_clima['Região'].isin(['Sudeste', 'Centro-Oeste'])].copy()
        df_clima_harmonizado = pd.concat([df_clima_harmonizado, df_clima_seco], ignore_index=True)
        print("✅ df_clima harmonizado.")
    except Exception as e:
        print(f"❌ ERRO ao harmonizar df_clima: {e}")
        return None, None
        
    return df_pop_harmonizado, df_clima_harmonizado


# --- 3. Função de Engenharia de Features ---

def create_features(df_consumo_agg, df_pop_harmonizado, df_clima_harmonizado):
    """Junta tudo e cria as features de Sazonalidade e Feriados."""
    print("Iniciando engenharia de features...")
    
    # --- Juntando (Merge) os dados ---
    # Seus dados de consumo são DIÁRIOS. Seus dados de clima são MENSAIS.
    # Vamos "espalhar" o dado mensal para todos os dias do respectivo mês.
    df_master = pd.merge(
        df_consumo_agg,
        df_clima_harmonizado[['Região', 'Ano', 'Mes_Num', 'Temperatura', 'Umidade']],
        on=['Região', 'Ano', 'Mes_Num'],
        how='left'
    )
    
    # Agora juntamos a população (que é ANUAL)
    df_master = pd.merge(
        df_master,
        df_pop_harmonizado[['Região', 'Ano', 'População 2024']],
        on=['Região', 'Ano'],
        how='left'
    )
    
    # --- Criando Features (Sazonalidade e Feriados) ---
    print("Criando features de Sazonalidade e Feriados...")
    
    # Sazonalidade (extraindo da data)
    df_master['dia_semana'] = df_master['Data'].dt.dayofweek  # 0=Segunda, 6=Domingo
    df_master['dia_mes'] = df_master['Data'].dt.day
    df_master['semana_ano'] = df_master['Data'].dt.isocalendar().week
    df_master['mes'] = df_master['Data'].dt.month
    df_master['trimestre'] = df_master['Data'].dt.quarter
    
    # Feriados (Usando a biblioteca 'holidays')
    # Pegando feriados nacionais do Brasil.
    # NOTA: Para um modelo ideal, incluiríamos feriados estaduais.
    feriados_br = holidays.country_holidays('BR')
    
    # Cria uma coluna 'e_feriado' (True se a data for feriado, False caso contrário)
    df_master['e_feriado'] = df_master['Data'].apply(lambda x: x in feriados_br)
    
    # Renomear colunas para o Prophet (ele exige 'ds' e 'y')
    df_master = df_master.rename(columns={
        'Data': 'ds',
        'Consumo_Limpo': 'y'
    })
    
    # Selecionar e ordenar colunas finais
    features_finais = [
        'ds', 'y', 'Região', 'Temperatura', 'Umidade', 'População 2024',
        'e_feriado', 'dia_semana', 'dia_mes', 'semana_ano', 'mes', 'trimestre'
    ]
    df_master_final = df_master[features_finais]
    
    print("✅ Master Dataset criado!")
    return df_master_final


# --- 4. Função Principal (Main) ---

def main():
    """Orquestra todo o pipeline de processamento de dados."""
    print("--- INICIANDO PIPELINE DE PROCESSAMENTO DE DADOS ---")
    
    # Passo 1: Carregar
    df_consumo, df_pop, df_clima = load_raw_data()
    if df_consumo is None:
        return

    # Passo 2: Limpar
    df_consumo_agg, df_pop, df_clima = clean_data(df_consumo, df_pop, df_clima)
    if df_consumo_agg is None:
        return

    # Passo 3: Harmonizar
    df_pop_harmonizado, df_clima_harmonizado = harmonize_regions(df_pop, df_clima)
    if df_pop_harmonizado is None:
        return

    # Passo 4: Criar Features e Master Dataset
    df_master = create_features(df_consumo_agg, df_pop_harmonizado, df_clima_harmonizado)
    if df_master is None:
        return

    # Passo 5: Salvar o resultado
    caminho_saida = DATA_PROCESSED_DIR / "master_dataset.csv"
    df_master.to_csv(caminho_saida, index=False, sep=';', decimal=',')
    
    print(f"\n--- SUCESSO! ---")
    print(f"Master Dataset salvo em: {caminho_saida}")
    print("Primeiras 5 linhas do dataset final:")
    print(df_master.head())


# --- Ponto de Entrada: Executa o 'main' se o script for chamado diretamente ---
if __name__ == "__main__":
    main()