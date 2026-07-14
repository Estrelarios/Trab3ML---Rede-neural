# Rascunho Relatório

Resumo

# Introdução

O trabalho é sobre aprendizado profundo

- Explicar o problema FONTE: TCC andre
- Explicar onde DeepLearning está inserido 
- Explicar SAC FONTE: TCC andre
    - Detalhar mais o actor e critic (MLPs)
- SAC, MLP e 2 camadas
    - Explicar porque MLPs do SAC pioram conforme aumentam as camadas ocultas -> ? (porque o anterior leva ao próximo?) -> passos muito grandes -> instabilidade -> explosão de gradiente -> Reta da Constante de Lipschitz quase vertical (ver depois)
    - Solução: Garantir que uma pequena entrada gere uma pequena saída e não uma grande -> Spectral Normalization.
- Spectral Normalization
    - Uma camada linear faz uma operação matemática simples: y = Wx (multiplica entrada x por matriz de pesos W) -> ReLu/Tahn/outro. Ex: 
    1.  Camada Linear 1  (Entrada -> Oculta 1) +  Tanh
    2.  Camada Linear 2  (Oculta 1 -> Oculta 2) +  Tanh
    3.  Camada Linear 3  (Oculta 2 -> Oculta 3) + Tanh
    4.  Camada Linear 4  (Oculta 3 -> Saída)
    - O fator máximo que a matriz W consegue esticar ou amplificar um vetor x é chamado de Maior Valor Singular (σₘₐₓ).
    - SN faz apenas uma coisa: a cada passo, calcula o σₘₐₓ da matriz de pesos e divide a matriz por ele. Ou seja, uma normalização 
    - Já implementado -> :D -> `torch.nn.utils.spectral_norm(nn.Linear(...))`


# Metodologia

- Detalhar enviroment
    - Action space - tabela presente em MuJuCo Humanoid-v5
    - Obs space - MuJuCo Humanoid-v5
    - Reward - padrão do enviroment MuJuCo Humanoid-v5

- Parâmetros do  melhor modelo

# Resultados e Conslusões

- Não escrever agora