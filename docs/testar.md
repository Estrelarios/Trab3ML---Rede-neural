# Ideias de parâmetros a testar (se der tempo)

Baseado no artigo `What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study`

# 3.2 Networks architecture (based on the results in Appendix E)

- Número de neurônios (largura) para as camadas do MLP. O artigo fala de largura do Value MLP e do Policy MLP (Qual a diferença?). Ver Figs 18 e 21.

# 3.6 Timesteps handling (based on the results in Appendix I)
- Fator de desconto = 0.999 pode melhorar um pouquinho o treino. Figs 60 e 61

# 3.7 Optimizers (based on the results in Appendix J)
- Adam melhor que RMSDrop
- LR de 0.0001 -> Fig 63
- Lr melhor com 0.9 de momentum (β1) (?) -> Fig 67