"""
Utilidades para carregar e normalizar datasets UCI.

Este módulo fornece funções para:
- Carregar configuração de datasets de arquivo YAML
- Buscar datasets do repositório UCI
- Aplicar one-hot encoding em features categóricas
- Normalizar features numéricas
- Binarizar targets
"""

from pathlib import Path
from typing import Dict, Tuple, Callable
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

try:
    from ucimlrepo import fetch_ucirepo
    UCI_AVAILABLE = True
except ImportError:
    UCI_AVAILABLE = False
    print("Warning: ucimlrepo not installed. Install with: pip install ucimlrepo")


def load_dataset_configs(config_path: str = "config/datasets_config.yaml") -> Dict:
    """
    Carrega configuração dos datasets do arquivo YAML.
    
    Args:
        config_path: Caminho para arquivo YAML com configuração dos datasets
        
    Returns:
        Dicionário com configurações dos datasets
        
    Raises:
        FileNotFoundError: Se o arquivo de configuração não existir
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Processar configurações e criar lambdas de binarização
    dataset_configs = {}
    for name, cfg in config['datasets'].items():
        binarize_lambda = create_binarize_lambda(
            cfg['binarize_type'], 
            cfg['binarize_value']
        )
        dataset_configs[name] = {
            'id': cfg['id'],
            'name': cfg['name'],
            'description': cfg['description'],
            'binarize_lambda': binarize_lambda,
            'binarize_type': cfg['binarize_type'],
            'binarize_value': cfg['binarize_value']
        }
    
    return dataset_configs


def create_binarize_lambda(binarize_type: str, value: int) -> Callable:
    """
    Cria função lambda para binarização baseada no tipo.
    
    Args:
        binarize_type: Tipo de binarização ('equal' ou 'greater')
        value: Valor de referência
        
    Returns:
        Função lambda para binarização
        
    Raises:
        ValueError: Se o tipo de binarização for inválido
    """
    if binarize_type == 'equal':
        return lambda s: (s == value).astype(int)
    elif binarize_type == 'greater':
        return lambda s: (s > value).astype(int)
    else:
        raise ValueError(f"Tipo de binarização inválido: {binarize_type}. Use 'equal' ou 'greater'.")


def load_and_preprocess_dataset(
    dataset_name: str,
    config: Dict,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega e preprocessa um dataset UCI.
    
    Aplica:
    - One-hot encoding em features categóricas
    - Normalização em features numéricas
    - Binarização do target
    - Remoção de NaNs
    
    Args:
        dataset_name: Nome do dataset
        config: Dicionário com configuração do dataset (id, binarize_lambda, etc.)
        verbose: Se True, imprime informações durante o processamento
        
    Returns:
        Tupla (X, y) com features normalizadas e target binarizado
        
    Raises:
        ImportError: Se ucimlrepo não estiver instalado
        Exception: Se houver erro ao carregar o dataset
    """
    if not UCI_AVAILABLE:
        raise ImportError("ucimlrepo não está instalado. Install with: pip install ucimlrepo")
    
    # Buscar dataset
    dataset = fetch_ucirepo(id=config['id'])
    
    # Separar features numéricas e categóricas
    X_numeric = dataset.data.features.select_dtypes(include=[np.number])
    X_categorical = dataset.data.features.select_dtypes(exclude=[np.number])
    
    y = dataset.data.targets.values.ravel()
    
    # Binarizar o target usando a lambda fornecida
    y_binary = config['binarize_lambda'](y)
    
    # Combinar features
    X_combined = pd.concat([X_numeric, X_categorical], axis=1)
    
    # Remover NaNs
    mask = ~(X_combined.isna().any(axis=1) | np.isnan(y_binary))
    X_clean = X_combined[mask]
    y_clean = y_binary[mask]
    
    # One-hot encoding nas categóricas
    if X_categorical.shape[1] > 0:
        if verbose:
            print(f"  → {dataset_name}: {X_categorical.shape[1]} features categóricas encontradas")
        
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_cat_encoded = encoder.fit_transform(X_clean[X_categorical.columns])
        
        # Combinar com numéricas
        X_num_array = X_clean[X_numeric.columns].values
        X_combined_array = np.hstack([X_num_array, X_cat_encoded])
    else:
        X_combined_array = X_clean[X_numeric.columns].values
    
    # Normalizar todas as features (numéricas + encoded categóricas)
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X_combined_array)
    
    if verbose:
        print(f"✓ {dataset_name} (UCI ID: {config['id']}) - {X_transformed.shape[0]} samples, {X_transformed.shape[1]} features")
    
    return X_transformed, y_clean


def load_all_datasets(
    config_path: str = "config/datasets_config.yaml",
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Carrega e preprocessa todos os datasets configurados no YAML.
    
    Args:
        config_path: Caminho para arquivo YAML com configuração dos datasets
        verbose: Se True, imprime informações durante o processamento
        
    Returns:
        Dicionário com nome do dataset como chave e tupla (X, y) como valor
    """
    datasets = {}
    
    if verbose:
        print("\nCarregando datasets...")
    
    # Carregar configurações
    dataset_configs = load_dataset_configs(config_path)
    
    # Carregar cada dataset
    for dataset_name, config in dataset_configs.items():
        try:
            X, y = load_and_preprocess_dataset(dataset_name, config, verbose)
            datasets[dataset_name] = (X, y)
        except Exception as e:
            if verbose:
                print(f"✗ {dataset_name} falhou: {e}")
    
    return datasets


# Função de conveniência para compatibilidade com código antigo
def get_datasets(config_path: str = "config/datasets_config.yaml") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Alias para load_all_datasets para compatibilidade com código existente.
    
    Args:
        config_path: Caminho para arquivo YAML com configuração dos datasets
        
    Returns:
        Dicionário com nome do dataset como chave e tupla (X, y) como valor
    """
    return load_all_datasets(config_path, verbose=True)
