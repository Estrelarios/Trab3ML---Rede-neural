import argparse
from pathlib import Path
from agent import SAC
from environment import make_env
from utils import load_config
from export import evaluate_agent

def compare_runs(runs_dir: Path, n_eval: int):
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

    print("\n" + "="*80)
    print(f"{'RANK':<5} | {'MODELO (PASTA)':<35} | {'RECOMPENSA MÉDIA':<20} | {'LR':<8}")
    print("="*80)
    
    for i, (name, mean, std, config) in enumerate(results, 1):
        lr = config.get("lr", "N/A")
        print(f"{i:<5} | {name:<35} | {mean:>8.2f} ± {std:<8.2f} | {lr}")
    
    print("="*80)

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
    args = parser.parse_args()

    compare_runs(Path(args.runs_dir), args.n_eval)
