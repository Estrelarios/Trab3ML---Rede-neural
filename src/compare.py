import argparse
import csv
from pathlib import Path
from agent import SAC
from environment import make_env
from utils import load_config
from export import evaluate_agent

def compare_runs(runs_dir: Path, n_eval: int, output_csv: str = "compare_results.csv"):
    """
    Itera sobre todas as pastas dentro de runs_dir, avalia o final_model.pt
    de cada uma, e imprime uma tabela comparativa de performance.
    """
    if not runs_dir.exists() or not runs_dir.is_dir():
        print(f"A pasta {runs_dir} não existe ou não é um diretório.")
        return

    results = []

    # Encontra todas as pastas de treino
    for run_folder in runs_dir.iterdir():
        if run_folder.is_dir():
            model_path = run_folder / "final_model.pt"
            config_path = run_folder / "config.yaml"

            if model_path.exists() and config_path.exists():
                print(f"Avaliando: {run_folder.name}...")
                
                # Carrega as configurações originais com as quais o modelo foi treinado
                config = load_config(config_path)
                env_name = config["env_name"]
                hidden_dim = config["hidden_dim"]
                seed = config.get("seed", 42)
                
                # Instancia o ambiente sem renderização gráfica (mais rápido)
                env, state_dim, action_dim = make_env(env_name, render_mode=None)
                
                # Instancia o Agente e carrega os pesos
                agent = SAC(state_dim, action_dim, hidden_dim, action_space=env.action_space)
                agent.load_model(model_path)
                
                # Avalia usando a mesma função oficial de exportação
                mean_reward, std_reward = evaluate_agent(agent, env, n_eval, seed)
                
                results.append((run_folder.name, mean_reward, std_reward, config))
            else:
                print(f"Ignorando {run_folder.name}: Falta final_model.pt ou config.yaml")

    # Ordena os resultados do melhor (maior recompensa) para o pior
    results.sort(key=lambda x: x[1], reverse=True)

    header_print = f"{'RANK':<5} | {'MODELO (PASTA)':<35} | {'RECOMPENSA MÉDIA':<20} | {'LR':<8} | {'HIDDEN_DIM':<15} | {'TOTAL_STEPS':<12} | {'BUFFER_SIZE':<12} | {'BATCH_SIZE':<10} | {'START_STEPS':<12} | {'GAMMA':<6} | {'TAU':<6} | {'ALPHA':<6}"
    separator = "=" * len(header_print)
    
    print("\n" + separator)
    print(header_print)
    print(separator)
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["RANK", "MODELO", "RECOMPENSA_MEDIA", "STD", "LR", "HIDDEN_DIM", "TOTAL_STEPS", "BUFFER_SIZE", "BATCH_SIZE", "START_STEPS", "GAMMA", "TAU", "ALPHA"])
        
        for i, (name, mean, std, config) in enumerate(results, 1):
            lr = config.get("lr", "N/A")
            hd = config.get("hidden_dim", "N/A")
            ts = config.get("total_steps", "N/A")
            bsz = config.get("buffer_size", "N/A")
            bch = config.get("batch_size", "N/A")
            ss = config.get("start_steps", "N/A")
            gm = config.get("gamma", "N/A")
            tau = config.get("tau", "N/A")
            alp = config.get("alpha", "N/A")
            
            print(f"{i:<5} | {name:<35} | {mean:>8.2f} ± {std:<8.2f} | {lr:<8} | {str(hd):<15} | {str(ts):<12} | {str(bsz):<12} | {str(bch):<10} | {str(ss):<12} | {str(gm):<6} | {str(tau):<6} | {str(alp):<6}")
            writer.writerow([i, name, mean, std, lr, hd, ts, bsz, bch, ss, gm, tau, alp])
    
    print(separator)
    print(f"\nResultados salvos com sucesso em: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara a performance de múltiplos agentes treinados.")
    parser.add_argument(
        "--runs-dir", 
        type=str, 
        default="runs", 
        help="Caminho para a pasta que contém as subpastas dos treinos (padrão: runs)"
    )
    parser.add_argument(
        "--n-eval", 
        type=int, 
        default=5, 
        help="Número de episódios para testar cada modelo (padrão: 5)"
    )
    parser.add_argument(
        "--output-csv", 
        type=str, 
        default="compare_results.csv", 
        help="Nome do arquivo CSV de saída (padrão: compare_results.csv)"
    )
    args = parser.parse_args()

    compare_runs(Path(args.runs_dir), args.n_eval, args.output_csv)
