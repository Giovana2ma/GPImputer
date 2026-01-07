# GP-based Imputation Project

Este projeto implementa um sistema de imputação de dados faltantes usando Programação Genética (GP) para combinar múltiplos métodos de imputação.

## Estrutura do Projeto

```
pocii/
├── config/                    # Arquivos de configuração
│   ├── experiment_config.yaml # Configuração de experimentos
│   └── gp_config.yaml        # Configuração do GP
├── data/                      # Módulo de dados
│   ├── __init__.py
│   ├── loaders.py            # Carregamento de datasets
│   └── missingness.py        # Geração de dados faltantes (MCAR, MAR, MNAR)
├── imputers/                  # Implementações de imputadores
│   ├── __init__.py
│   ├── base.py               # Classe base
│   ├── simple.py             # Mean, Median, Mode
│   ├── knn_imputer.py        # KNN
│   ├── mice_imputer.py       # MICE
│   ├── missforest_imputer.py # MissForest
│   └── svd_imputer.py        # SVD/Matrix Factorization
├── gp/                        # Módulo de Programação Genética
│   ├── __init__.py
│   ├── operators.py          # Operadores protegidos
│   ├── primitives.py         # Setup de primitivas DEAP
│   ├── fitness.py            # Funções de fitness
│   └── gp_imputer.py         # Imputador GP principal
├── evaluation/                # Avaliação e testes
│   ├── __init__.py
│   ├── metrics.py            # RMSE, MAE, NRMSE, R²
│   ├── statistical_tests.py  # Wilcoxon, Friedman, Nemenyi
│   └── visualization.py      # Plots e gráficos
├── experiments/               # Scripts de experimentos
│   ├── run_experiments.py    # Script principal
│   └── utils.py              # Utilitários
├── notebooks/                 # Notebooks Jupyter para análise
├── logs/                      # Logs de execução
├── results/                   # Resultados dos experimentos
└── requirements.txt           # Dependências

```

## Instalação

### 1. Criar ambiente virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## Uso

### Executar Experimentos

```bash
cd /scratch/giovana.assis/pocii
python experiments/run_experiments.py
```

Com configurações customizadas:

```bash
python experiments/run_experiments.py \
    --config config/experiment_config.yaml \
    --gp-config config/gp_config.yaml
```

### Configuração

#### Experiment Config (`config/experiment_config.yaml`)

- **Datasets**: Define quais datasets usar (sklearn ou custom)
- **Missingness**: Mecanismos (MCAR, MAR, MNAR), taxas (5%, 10%, 20%, etc.), seeds
- **Imputers**: Quais imputadores habilitar e seus parâmetros
- **Evaluation**: Métricas e testes estatísticos
- **Experiment**: Número de repetições, paths de saída
- **Cross-Validation**: Habilitar/desabilitar validação cruzada e número de folds

##### Ativando Cross-Validation

No arquivo `config/experiment_config.yaml`:

```yaml
experiment:
  use_cross_validation: true  # Habilita validação cruzada
  n_folds: 5                  # Número de folds (padrão: 5)
```

Com cross-validation habilitada, o dataset é dividido em `n_folds` partes. Cada fold é usado como conjunto de teste, e os resultados são agregados (média e desvio padrão) automaticamente.

#### GP Config (`config/gp_config.yaml`)

- **Population**: Tamanho da população, gerações, elitismo
- **Tree**: Profundidade máxima, tamanho máximo, método de inicialização
- **Evolution**: Taxas de crossover/mutação, tournament size
- **Functions**: Conjunto de funções (operadores)
- **Terminals**: Conjunto de terminais (constantes, imputadores)
- **Fitness**: Métrica e penalização por complexidade

##### Fitness Baseado em Classificador

O GP pode otimizar diretamente para o desempenho de um classificador (F1-score) ao invés de métricas de imputação:

```yaml
fitness:
  metric: "f1_classifier"  # Usa F1-score do classificador como fitness
  parsimony_coefficient: 0.0001
  
  classifier:
    type: "random_forest"  # random_forest, logistic_regression, svm, decision_tree, knn
    params:
      n_estimators: 100
      max_depth: 10
      random_state: 42
  cv_folds: 5  # Cross-validation folds para avaliar o classificador
```

Quando `metric: "f1_classifier"` está configurado:
- O GP evolui programas que maximizam o F1-score do classificador especificado
- A imputação é avaliada treinando o classificador nos dados imputados
- Cross-validation é usado para avaliar o classificador de forma robusta
- O fitness é `1 - F1_score` (minimização)

## Componentes Principais

### 1. Imputadores Base

Todos os imputadores seguem a interface comum (`fit`, `transform`, `fit_transform`):

- **MeanImputer**: Imputação por média
- **MedianImputer**: Imputação por mediana
- **KNNImputerWrapper**: K-Nearest Neighbors
- **MICEImputerWrapper**: Multivariate Imputation by Chained Equations
- **MissForestImputerWrapper**: Random Forest iterativo
- **SVDImputerWrapper**: Decomposição SVD

### 2. GP Imputer

O `GPImputer` evolui programas (árvores) que combinam as saídas dos imputadores base usando:

**Arquitetura**: **Multi-Árvore Integrada**
- **1 algoritmo GP** é executado
- Cada indivíduo contém **N árvores** (uma por feature)
- Fitness calculado **globalmente** no dataset completo
- **N× mais rápido** que evoluções independentes
- Features **co-evoluem** para otimização integrada

**Function Set** (protegido contra erros numéricos):
- Binários: `+, -, *, /, min, max, pow`
- Unários: `sqrt, log, exp, abs`
- Ternário: `if-then-else`

**Terminal Set**:
- Saídas dos imputadores base (para a feature específica)
- Constantes fixas: `[-10, -1, 0, 1, 2, 10]`
- Constantes efêmeras: valores aleatórios em `[-10, 10]`

**Fitness**: 
- **Métricas de Imputação**: NRMSE, RMSE, MAE (avaliado em todas as features)
- **Métricas de Classificação**: F1-score de classificador (avaliado no dataset completo)

### 3. Geração de Missingness

- **MCAR** (Missing Completely At Random): Remoção aleatória uniforme
- **MAR** (Missing At Random): Dependente de outras variáveis observadas
- **MNAR** (Missing Not At Random): Dependente do próprio valor faltante

### 4. Avaliação

**Métricas**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- NRMSE (Normalized RMSE)
- R² (Coefficient of Determination)

**Testes Estatísticos**:
- Wilcoxon signed-rank (comparação pareada)
- Friedman test (comparação múltipla)
- Nemenyi post-hoc test

## Resultados

Os resultados são salvos em `results/experiment_YYYYMMDD_HHMMSS/`:

- `results.pkl`: Resultados completos (pickle)
- `summary.csv`: Resumo em formato tabular
  - Com CV: inclui colunas `*_std` com desvio padrão de cada métrica
  - Sem CV: apenas valores de cada execução individual
- `statistical_tests.json`: Resultados dos testes estatísticos

### Histórico de Evolução do GP

Cada execução do GP salva automaticamente o histórico completo em `results/gp_history/`:

- **Arquivo**: `gp_evolution_YYYYMMDD_HHMMSS.json.gz` (compactado)
- **Conteúdo**:
  - Fitness de todos os indivíduos em todas as gerações
  - Programas (árvores) de todos os indivíduos
  - Tamanho das árvores
  - Metadados da execução
  - Melhor indivíduo final

**Análise do Histórico**:
```bash
# Analisar histórico mais recente
python examples/analyze_gp_history.py
```

**Carregar Histórico Programaticamente**:
```python
from gp import GPImputer

# Carregar histórico
history = GPImputer.load_evolution_history('results/gp_history/gp_evolution_20251118_120000.json.gz')

# Analisar
analysis = GPImputer.analyze_history('results/gp_history/gp_evolution_20251118_120000.json.gz')

# Acessar dados
print(analysis['convergence'])  # Fitness por geração
print(analysis['diversity'])     # Diversidade por geração
print(analysis['final_best'])    # Melhor indivíduo
```

### Estrutura dos Resultados com Cross-Validation

Quando cross-validation está habilitada, os resultados incluem:

```python
{
  'dataset_name': {
    'MCAR': {
      0.1: {  # missing_rate
        42: {  # seed
          'folds': [fold_0_results, fold_1_results, ...],
          'aggregated': {
            'method_name': {
              'metrics': {
                'rmse_mean': 0.123,
                'rmse_std': 0.012,
                'mae_mean': 0.098,
                'mae_std': 0.009,
                ...
              }
            }
          }
        }
      }
    }
  }
}
```

## Exemplos de Uso

### Scripts de Exemplo Completos

Exemplos práticos disponíveis em `examples/`:

```bash
# Demonstra GP com fitness de classificador
python examples/gp_with_classifier_fitness.py

# Compara fitness de imputação vs. fitness de classificador
python examples/compare_fitness_approaches.py

# Demonstra uma árvore GP por feature
python examples/per_feature_trees.py

# Demonstra arquitetura multi-árvore
python examples/multitree_demo.py

# Analisa histórico de evolução do GP
python examples/analyze_gp_history.py
```

Estes scripts demonstram:
- Criação de dados com missing values
- Treinamento de imputadores base
- Uso do GP com diferentes configurações de fitness
- Uma árvore GP personalizada por feature
- Análise completa do histórico de evolução
- Comparação de resultados

---

## 🔍 Otimização de Hiperparâmetros (Optuna)

### Otimizar Imputadores Automaticamente

```bash
# Executar otimização com Optuna
python experiments/optuna_optimization.py --config config/optuna_config.yaml

# Analisar resultados
python experiments/analyze_optuna_results.py results/optuna_optimization/all_optimization_results_*.csv
```

**Recursos:**
- ✅ Otimização automática de KNN, MICE, MissForest, SVD
- ✅ Múltiplos datasets e níveis de missing values
- ✅ Paralelização de trials
- ✅ Visualizações interativas (Plotly)
- ✅ Persistência em SQLite

**Configuração rápida:**

```yaml
# config/optuna_config.yaml
optuna:
  n_trials: 100
  n_jobs: 4

datasets:
  - name: "breast_cancer"
    type: "sklearn"

imputers:
  knn:
    enabled: true
    params:
      n_neighbors: {type: "int", low: 3, high: 20}
      weights: {type: "categorical", choices: ["uniform", "distance"]}
```

📖 **Documentação completa:** [docs/OPTUNA_OPTIMIZATION.md](docs/OPTUNA_OPTIMIZATION.md)

### Uso Individual de Imputadores

```python
from imputers import MeanImputer, KNNImputerWrapper
import numpy as np

# Dados com valores faltantes
X = np.array([[1, 2, np.nan],
              [4, np.nan, 6],
              [7, 8, 9]])

# Mean Imputer
mean_imp = MeanImputer()
X_mean = mean_imp.fit_transform(X)

# KNN Imputer
knn_imp = KNNImputerWrapper(n_neighbors=2)
X_knn = knn_imp.fit_transform(X)
```

### Uso do GP Imputer

#### Modo 1: Fitness baseado em métricas de imputação

```python
from gp import GPImputer
from imputers import MeanImputer, MedianImputer

# Preparar base imputers
base_imputers = {
    'imp_mean': MeanImputer().fit(X),
    'imp_median': MedianImputer().fit(X)
}

# GP config
gp_config = {
    'population': {'size': 100, 'generations': 30},
    'tree': {'max_depth': 5},
    'fitness': {'metric': 'rmse', 'parsimony_coefficient': 0.001}
}

# Criar e treinar GP
gp_imp = GPImputer(config=gp_config)
gp_imp.fit(X, base_imputers, y_true=X_complete[missing_mask])

# Imputar
X_gp = gp_imp.transform(X)

# Ver melhor programa
print(gp_imp.get_best_program())
```

#### Modo 2: Fitness baseado em classificador (F1-score)

```python
from gp import GPImputer
from imputers import MeanImputer, MedianImputer, KNNImputerWrapper
from sklearn.ensemble import RandomForestClassifier

# Preparar base imputers
base_imputers = {
    'imp_mean': MeanImputer().fit(X),
    'imp_median': MedianImputer().fit(X),
    'imp_knn': KNNImputerWrapper(n_neighbors=5).fit(X)
}

# GP config com fitness de classificador
gp_config = {
    'population': {'size': 100, 'generations': 30},
    'tree': {'max_depth': 5},
    'fitness': {
        'metric': 'f1_classifier',
        'classifier': {
            'type': 'random_forest',
            'params': {'n_estimators': 100, 'random_state': 42}
        },
        'cv_folds': 5
    }
}

# Criar classificador
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Criar e treinar GP (precisa de y_target para classificação)
gp_imp = GPImputer(config=gp_config)
gp_imp.fit(X, base_imputers, classifier=classifier, y_target=y_labels)

# Imputar
X_gp = gp_imp.transform(X)

# Ver melhores programas
programs = gp_imp.get_best_program()
print(f"Evolved {len(programs)} GP trees, one per feature")
for feat_name, prog in list(programs.items())[:3]:  # Show first 3
    print(f"{feat_name}: {prog}")
print(f"Best fitness (1 - F1): {gp_imp.best_fitness_:.4f}")
```

### Gerar Dados Faltantes

```python
from data import generate_missing_data

X_complete = np.random.rand(100, 10)

# MCAR
X_mcar, mask_mcar = generate_missing_data(X_complete, 'MCAR', 0.2, seed=42)

# MAR
X_mar, mask_mar = generate_missing_data(X_complete, 'MAR', 0.2, seed=42,
                                       dependency_col=0)

# MNAR
X_mnar, mask_mnar = generate_missing_data(X_complete, 'MNAR', 0.2, seed=42)
```

## Extensões Futuras

- [ ] Suporte para dados categóricos
- [ ] Otimização de hiperparâmetros dos base imputers
- [ ] Ensemble de múltiplos programas GP
- [ ] Visualizações interativas (Plotly Dash)
- [ ] Paralelização distribuída (Dask, Ray)
- [x] ✅ Cross-validation integrada
- [x] ✅ Fitness baseado em classificador (F1-score)
- [x] ✅ Arquitetura multi-árvore integrada (1 GP, N árvores por indivíduo)

## Documentação Adicional

- **[Guia de Fitness de Classificador](docs/CLASSIFIER_FITNESS_GUIDE.md)**: Documentação completa sobre fitness baseado em classificador
- **[Arquitetura Multi-Árvore](docs/MULTITREE_ARCHITECTURE.md)**: Explicação detalhada da arquitetura multi-árvore integrada
- **Exemplos práticos**:
  - `examples/gp_with_classifier_fitness.py`: Uso básico do GP com classificador
  - `examples/compare_fitness_approaches.py`: Comparação entre abordagens de fitness
  - `examples/per_feature_trees.py`: Demonstração de árvores personalizadas por feature

## Referências

- DEAP: Distributed Evolutionary Algorithms in Python
- Scikit-learn: Machine Learning in Python
- Programação Genética para imputação de dados

## Licença

MIT License

## Autores

Projeto desenvolvido para pesquisa em imputação de dados usando Programação Genética.
