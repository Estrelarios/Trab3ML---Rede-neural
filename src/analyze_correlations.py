import pandas as pd
import numpy as np
import argparse

def analyze_spearman(csv_path: str):
    print(f"Lendo arquivo: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{csv_path}' não encontrado.")
        return

    # Substitui strings 'N/A' por NaN e ignora modelo/rank
    df = df.replace('N/A', np.nan)
    
    colunas_hiperparametros = [
        'LR', 'HIDDEN_DIM', 'TOTAL_STEPS', 'BUFFER_SIZE', 
        'BATCH_SIZE', 'START_STEPS', 'GAMMA', 'TAU', 'ALPHA'
    ]
    
    # Força conversão para float
    for col in colunas_hiperparametros + ['RECOMPENSA_MEDIA']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("\nCalculando Correlação de Spearman com RECOMPENSA_MEDIA...\n")
    
    # Calcula a matriz de correlação usando pandas (método spearman)
    corr_matrix = df.corr(method='spearman')
    
    if 'RECOMPENSA_MEDIA' not in corr_matrix.columns:
        print("Erro: Coluna 'RECOMPENSA_MEDIA' não encontrada ou sem dados numéricos.")
        return
        
    # Isola o target, removendo a própria recompensa, desvio padrão e rank
    recompensa_corr = corr_matrix['RECOMPENSA_MEDIA'].drop(labels=['RECOMPENSA_MEDIA', 'STD', 'RANK'], errors='ignore')
    
    # Ordena do maior impacto positivo ao maior negativo
    recompensa_corr = recompensa_corr.sort_values(ascending=False)
    
    print("="*65)
    print(f"{'HIPERPARÂMETRO':<18} | {'CORRELAÇÃO (SPEARMAN)':<22} | {'IMPACTO'}")
    print("="*65)
    
    for param, score in recompensa_corr.items():
        if pd.isna(score):
            impact = "INCONCLUSIVO (Sem variação de dados)"
            score_str = "N/A"
        else:
            score_str = f"{score:.4f}"
            if score > 0.5:
                impact = "FORTE POSITIVO 🟩"
            elif score > 0.2:
                impact = "FRACO POSITIVO 🟨"
            elif score >= -0.2:
                impact = "NEUTRO ⬜"
            elif score >= -0.5:
                impact = "FRACO NEGATIVO 🟧"
            else:
                impact = "FORTE NEGATIVO 🟥"
                
        print(f"{param:<18} | {score_str:>22} | {impact}")
    
    print("="*65)
    print("\nComo ler:")
    print("-> Próximo de +1: Quanto MAIOR o parâmetro, MAIOR tendeu a ser a recompensa.")
    print("-> Próximo de -1: Quanto MAIOR o parâmetro, MENOR tendeu a ser a recompensa.")
    print("-> Próximo de  0: Parâmetro não apresentou tendência de alta ou baixa direta.")
    print("-> N/A: O hiperparâmetro foi idêntico em todos os treinos (sem variância).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisa a correlação de Spearman de um CSV gerado.")
    parser.add_argument("--csv", type=str, default="compare_results.csv", help="Caminho do arquivo CSV")
    args = parser.parse_args()
    
    analyze_spearman(args.csv)
