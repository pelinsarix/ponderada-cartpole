## Vídeo Explicativo

(link)

# Otimização de Parâmetros para Q-Learning no Problema CartPole

Este projeto implementa uma solução para o problema CartPole (`CartPole-v1`) utilizando Q-learning com otimização de parâmetros. O código permite testar diferentes combinações de hiperparâmetros para encontrar a configuração ideal que maximize o desempenho do agente.

## Modelagem do Problema como MDP

O problema CartPole é modelado como um Processo de Decisão de Markov (MDP) com os seguintes componentes:

- **Estados (S)**: A observação contínua do ambiente composta por 4 dimensões:
  - Posição do carrinho (entre -4.8 e 4.8)
  - Velocidade do carrinho
  - Ângulo do pêndulo (entre -0.418 e 0.418 radianos)
  - Velocidade angular do pêndulo

- **Ações (A)**: O agente pode escolher entre duas ações:
  - 0: Empurrar o carrinho para a esquerda
  - 1: Empurrar o carrinho para a direita

- **Recompensas (R)**: O ambiente fornece +1 de recompensa para cada passo em que o pêndulo não cai.

- **Transições (P)**: A dinâmica do sistema físico determina como o estado muda após cada ação.

- **Fator de Desconto (γ)**: Valoriza recompensas futuras quase tanto quanto recompensas imediatas.

## Algoritmo Q-Learning

O Q-learning é um método de aprendizado por diferença temporal (TD) que aprende diretamente a função de valor de ação ótima Q*, independentemente da política seguida. A atualização da função Q é baseada na equação de Bellman:

```
Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
```

Onde:
- Q(s,a) é o valor atual do par estado-ação
- α é a taxa de aprendizado
- r é a recompensa recebida
- γ é o fator de desconto
- max_a' Q(s',a') é o valor da melhor ação no próximo estado
- [r + γ * max_a' Q(s',a') - Q(s,a)] é o erro TD

## Otimização de Parâmetros

O código implementa uma busca em grade (grid search) para testar diferentes combinações de parâmetros:

- **alpha (α)**: Taxa de aprendizado
- **gamma (γ)**: Fator de desconto
- **epsilon_initial**: Valor inicial para exploração
- **epsilon_decay**: Taxa de decaimento do epsilon
- **epsilon_min**: Valor mínimo de epsilon

Para cada combinação de parâmetros, o agente é treinado e avaliado, e os resultados são registrados. No final, o código identifica a melhor configuração e treina um modelo final com essa configuração.

## Como Usar

### 1. Definir os Parâmetros a Serem Testados

Modifique o dicionário `param_grid` para incluir os valores que você deseja testar:

```python
param_grid = {
    'alpha': [0.05, 0.1, 0.2],
    'gamma': [0.95, 0.99, 0.999],
    'epsilon_initial': [0.5, 1.0, 5.0],  
    'epsilon_decay': [0.99, 0.995, 0.998],
    'epsilon_min': [0.001, 0.01, 0.05] 
}
```

### 2. Ajustar os Parâmetros da Busca

Você pode ajustar o número de episódios para treinamento e avaliação:

```python
results_df, best_params, best_Q = grid_search_parameters(
    param_grid, 
    num_episodes=1000, 
    num_eval_episodes=5 
)
```

### 3. Executar o Código

Execute o notebook ou script. O código irá:
1. Testar todas as combinações de parâmetros
2. Exibir os resultados das melhores configurações
3. Treinar um modelo final com os melhores parâmetros
4. Visualizar o desempenho do modelo final
5. Salvar os resultados em um arquivo CSV

## Resultados

O código gera os seguintes resultados:

1. **Tabela de Parâmetros**: Uma tabela com as melhores configurações de parâmetros.
2. **Gráficos de Desempenho**: Gráficos mostrando o desempenho do treinamento final.
3. **Visualização da Política**: Uma animação mostrando o comportamento do agente treinado.
4. **Arquivo CSV**: Os resultados completos da busca de parâmetros são salvos para análise posterior.

## Comparação com Deep Q-Network (DQN)

Se fosse permitido o uso de redes neurais, uma abordagem usando Deep Q-Network (DQN) seria mais adequada para este problema pelos seguintes motivos:

1. **Generalização**: Uma rede neural pode generalizar melhor entre estados similares, sem a necessidade de discretização que pode perder informações importantes.

2. **Representação contínua**: O DQN poderia trabalhar diretamente com o espaço de estados contínuo, capturando nuances que são perdidas na discretização.

3. **Escalabilidade**: Para discretizações mais finas (com mais bins), a tabela Q se torna muito grande e sofre com o problema da "maldição da dimensionalidade", enquanto redes neurais escalam melhor.

4. **Aprendizado de características**: Uma rede neural poderia identificar automaticamente quais características do estado são mais relevantes para a decisão.

Em uma implementação com DQN, utilizaria:
- Uma rede neural com camadas totalmente conectadas
- Replay buffer para armazenar e reutilizar experiências passadas
- Rede alvo para estabilizar o aprendizado
- Normalização das entradas para melhorar a convergência

## Referências

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
- OpenAI Gym CartPole Documentation: https://www.gymlibrary.dev/environments/classic_control/cart_pole/