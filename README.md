# SAC MuJoCo Humanoid

## Resumo
Projeto de Deep Reinforcement Learning focado na resolução do ambiente de controle contínuo `Humanoid-v4` (MuJoCo/Gymnasium). O projeto implementa o algoritmo **Soft Actor-Critic (SAC)** modificado com base em achados de estudos empíricos de larga escala e estado-da-arte, como a aplicação de **Spectral Normalization** localizada no Crítico e técnicas de inicialização de pesos (`zero-mean actions`) no Ator para garantir a convergência em arquiteturas profundas e evitar o colapso por perdas de plasticidade matemática.

## Estrutura do Projeto
```text
Trab3ML---Rede-neural/
├── pyproject.toml         # Definições do ambiente isolado (Poetry)
├── poetry.lock            # Árvore de dependências resolvidas
├── main.py                # Ponto de entrada (Script de execução principal)
├── humanoide.py           # Script auxiliar de ambiente
└── sac_mujoco_humanoid/   # Núcleo da implementação RL
    ├── model.py           # Redes Neurais (Ator e Crítico + Spectral Norm)
    ├── agent.py           # Lógica do Algoritmo SAC (Perdas, Updates, Otimizadores)
    ├── train.py           # Loop de iteração (Interação com ambiente vs Otimização)
    ├── buffer.py          # Memória de Experiência (Replay Buffer)
    ├── environment.py     # Wrap do Gymnasium
    └── utils.py           # Utilitários e Logs
```

## Como configurar e ativar o ambiente

O projeto usa o gerenciador de pacotes **Poetry** (presume-se Python 3.13 já instalado no sistema).

### Windows

1. **Abra o terminal na pasta do projeto** (PowerShell recomendado):
```powershell
cd C:\Users\mathe\Documentos\CiênciaDaComputação\Graduacao\4Ano\AM\Trab3\Trab3ML---Rede-neural
```

2. **Inicialize o ambiente e instale as dependências:**
Isso forçará o uso do Python 3.13, lerá o arquivo `poetry.lock` e baixará a versão correta do PyTorch, Gymnasium com MuJoCo, etc.
```powershell
poetry env use python3.13
poetry install
```

3. **Ative o ambiente virtual:**
Isso vai "entrar" na bolha do projeto onde as bibliotecas foram instaladas. O prefixo do seu PowerShell vai mudar para indicar o ambiente ativo.
```powershell
poetry shell
```

### Linux

1. **Abra o terminal na pasta do repositório clonado:**
```bash
cd /caminho/para/Trab3ML---Rede-neural
```

2. **Inicialize o ambiente e instale as dependências:**
Isso forçará o uso do Python 3.13 e instalará todas as dependências do projeto.
```bash
poetry env use python3.13
poetry install
```

3. **Ative o shell do ambiente virtual:**
```bash
eval $(poetry env activate)
```

## Como Usar (Features)

Para executar qualquer funcionalidade do projeto (treino, visualização ou exportação), certifique-se de estar com o ambiente virtual ativado e **navegue para a pasta do código-fonte onde está o `config.yaml`**:

```powershell
cd sac_mujoco_humanoid
```

### Treinamento
Para iniciar o treinamento do agente SAC com os hiperparâmetros padrões:
*(Atenção: O Replay Buffer pré-aloca memória. Garanta ter pelo menos 8GB de RAM livre).*
```powershell
python -m train
```

### Configurações
Edite o arquivo `config.yaml` dentro de `sac_mujoco_humanoid` para alterar tamanho da rede, passos (total_steps), *batch size*, *learning rate* ou semente (seed).

### Visualização (Enjoy)
Para assistir um agente já treinado executando no ambiente, rode o script `enjoy` apontando para o arquivo salvo do modelo na pasta `runs/`:
```powershell
python -m enjoy --artifact runs/sac_Humanoid-v5_.../final_model.pt --num-episodes 5
```

### Exportação (Hugging Face)
Gera automaticamente um *Model Card*, grava um vídeo do agente no ambiente e sobe os pesos para o Hugging Face Hub:
```powershell
python -m export \
    --username SEU_USUARIO_HF \
    --repo-name sac-mujoco-humanoid \
    --artifact-path runs/sac_Humanoid-v5_.../final_model.pt \
    --movie-fps 30 \
    --n-eval 10
```
