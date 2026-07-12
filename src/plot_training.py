import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_history(run_folder: str, show_plot: bool = True):
    csv_path = Path(run_folder) / "training_history.csv"
    
    if not csv_path.exists():
        print(f"Erro: O arquivo '{csv_path}' não foi encontrado.")
        return
        
    print(f"Lendo dados de {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if df.empty or 'Step' not in df.columns or 'Reward' not in df.columns:
        print("Erro: CSV está vazio ou com cabeçalhos inválidos.")
        return
        
    # Plotagem
    plt.figure(figsize=(10, 6))
    
    # Linha principal transparente pros episódios brutos
    plt.plot(df['Step'], df['Reward'], alpha=0.3, color='royalblue', label='Recompensa Bruta')
    
    # Média móvel para alisar a curva e ver a tendência geral 
    window_size = min(20, max(1, len(df) // 10))
    rolling_mean = df['Reward'].rolling(window=window_size, min_periods=1).mean()
    plt.plot(df['Step'], rolling_mean, linewidth=2, color='darkblue', label=f'Média Móvel ({window_size} eps)')
    
    plt.title("Curva de Aprendizado (Recompensa vs Steps)", fontsize=14)
    plt.xlabel("Steps de Treinamento", fontsize=12)
    plt.ylabel("Recompensa do Episódio", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper left")
    
    # Salvar a imagem na mesma pasta do modelo
    output_img = Path(run_folder) / "learning_curve.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo com sucesso em: {output_img}")
    
    # Tenta exibir na tela se solicitado
    if show_plot:
        try:
            plt.show()
        except Exception:
            pass
            
    plt.close()

def plot_all_histories(runs_dir: str):
    runs_path = Path(runs_dir)
    if not runs_path.exists() or not runs_path.is_dir():
        print(f"Erro: O diretório '{runs_dir}' não existe.")
        return
        
    print(f"Analisando todas as subpastas em '{runs_dir}'...")
    for run_folder in runs_path.iterdir():
        if run_folder.is_dir():
            csv_path = run_folder / "training_history.csv"
            if csv_path.exists():
                plot_history(str(run_folder), show_plot=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plota o gráfico de treinamento a partir do histórico de uma run.")
    parser.add_argument("--run-dir", type=str, help="Caminho para uma pasta específica da run (ex: runs/sac_Humanoid_XX)")
    parser.add_argument("--all-runs", type=str, help="Caminho para a pasta principal que contém todas as runs (ex: runs/)")
    args = parser.parse_args()
    
    if args.all_runs:
        plot_all_histories(args.all_runs)
    elif args.run_dir:
        plot_history(args.run_dir)
    else:
        print("Erro: Você deve fornecer --run-dir ou --all-runs.")
