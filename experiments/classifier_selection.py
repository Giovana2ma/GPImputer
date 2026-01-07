"""
Script para seleção do melhor classificador.

Testa diferentes classificadores em múltiplos datasets e analisa o tradeoff
entre F1-score e tempo de execução para escolher o melhor para fitness do GP.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Tuple
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, StratifiedKFold
import importlib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset_loader import load_dataset_configs, load_and_preprocess_dataset
from analysis.classifier_analysis import generate_all_plots, print_analysis_summary

sns.set_style("whitegrid")


class ClassifierSelector:
    """Classe para seleção e avaliação de classificadores."""
    
    def __init__(self, results_dir: str = "results/classifier_selection", 
                 datasets_config: str = "config/datasets_config.yaml",
                 classifiers_config: str = "config/classifiers_config.yaml"):
        """
        Inicializa o seletor de classificadores.
        
        Args:
            results_dir: Diretório para salvar resultados
            datasets_config: Caminho para arquivo YAML com configuração dos datasets
            classifiers_config: Caminho para arquivo YAML com configuração dos classificadores
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.datasets_config_path = datasets_config
        self.classifiers_config_path = classifiers_config
        self.dataset_configs = load_dataset_configs(datasets_config)
        
    def _load_classifier_class(self, class_path: str):
        """
        Carrega uma classe de classificador dinamicamente.
        
        Args:
            class_path: Caminho completo da classe (e.g., 'sklearn.ensemble.RandomForestClassifier')
            
        Returns:
            Classe do classificador
        """
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    def get_classifiers(self) -> Dict:
        """
        Carrega os classificadores a partir do arquivo YAML.
        
        Returns:
            Dicionário com nome e instância do classificador
        """
        config_path = Path(self.classifiers_config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Arquivo de configuração de classificadores não encontrado: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        classifiers = {}
        
        for name, clf_config in config['classifiers'].items():
            # Pular classificadores opcionais se não estiverem disponíveis
            if clf_config.get('optional', False):
                try:
                    clf_class = self._load_classifier_class(clf_config['type'])
                    classifiers[name] = clf_class(**clf_config['params'])
                except (ImportError, ModuleNotFoundError) as e:
                    print(f"Warning: Classificador opcional '{name}' não disponível: {e}")
                    continue
            else:
                # Classificadores obrigatórios
                clf_class = self._load_classifier_class(clf_config['type'])
                classifiers[name] = clf_class(**clf_config['params'])
        
        return classifiers
        
    
    def get_datasets(self) -> Dict:
        """
        Carrega e normaliza os datasets (sklearn e UCI).
        One-hot encoding para features categóricas.
        
        Returns:
            Dicionário com nome e (X, y) do dataset
        """
        datasets = {}
        
        print("\nCarregando datasets...")
        
        for dataset_name, config in self.dataset_configs.items():
            try:
                X, y = load_and_preprocess_dataset(dataset_name, config, verbose=True)
                datasets[dataset_name] = (X, y)
            except Exception as e:
                print(f"✗ {dataset_name} falhou: {e}")
        
        return datasets
    
    def evaluate_classifier(
        self,
        clf_name: str,
        clf,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Tuple[float, float, float]:
        """
        Avalia um classificador em um dataset.
        
        Args:
            clf_name: Nome do classificador
            clf: Instância do classificador
            X: Features
            y: Target
            cv: Número de folds para cross-validation
            missing_ratio: Percentual de missing values
            
        Returns:
            (f1_score_mean, f1_score_std, execution_time)
        """

        
        # Cross-validation com tempo
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        start_time = time.time()
        scores = cross_val_score(
            clf,
            X,
            y,
            cv=cv_splitter,
            scoring='f1_weighted',
            n_jobs=1  # Para medir tempo corretamente
        )
        execution_time = time.time() - start_time
        
        return scores.mean(), scores.std(), execution_time
    
    def run_evaluation(self):
        """Executa avaliação completa de todos classificadores e datasets."""
        print("="*70)
        print("SELEÇÃO DE CLASSIFICADOR PARA FITNESS DO GP")
        print("="*70)
        
        classifiers = self.get_classifiers()
        datasets = self.get_datasets()
        
        total_combinations = len(classifiers) * len(datasets)
        current = 0
        
        print(f"\nTestando {len(classifiers)} classificadores em {len(datasets)} datasets")
        print(f"Total de avaliações: {total_combinations}\n")
        
        # Testar cada combinação
        for dataset_name, (X, y) in datasets.items():
            print(f"\n{'='*70}")
            print(f"Dataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
            print(f"{'='*70}")
            
            for clf_name, clf in classifiers.items():
                current += 1
                print(f"[{current}/{total_combinations}] Testando {clf_name}...", end=' ')
                
                try:
                    f1_mean, f1_std, exec_time = self.evaluate_classifier(
                        clf_name, clf, X, y
                    )
                    
                    result = {
                        'dataset': dataset_name,
                        'classifier': clf_name,
                        'f1_score_mean': f1_mean,
                        'f1_score_std': f1_std,
                        'execution_time': exec_time,
                        'n_samples': X.shape[0],
                        'n_features': X.shape[1],
                        'n_classes': len(np.unique(y))
                    }
                    
                    self.results.append(result)
                    
                    print(f"F1={f1_mean:.4f}±{f1_std:.4f}, Time={exec_time:.2f}s")
                    
                except Exception as e:
                    print(f"ERRO: {e}")
        
        print("\n" + "="*70)
        print("Avaliação concluída!")
        print("="*70)
    
    def save_results(self):
        """Salva resultados em CSV."""
        df = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_path = self.results_dir / f"classifier_evaluation_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResultados salvos em: {csv_path}")
        
        return df, csv_path


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Seleção de classificador para GP')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/classifier_selection',
        help='Diretório para salvar resultados'
    )
    parser.add_argument(
        '--datasets-config',
        type=str,
        default='config/datasets_config.yaml',
        help='Arquivo YAML com configuração dos datasets'
    )
    parser.add_argument(
        '--classifiers-config',
        type=str,
        default='config/classifiers_config.yaml',
        help='Arquivo YAML com configuração dos classificadores'
    )
    parser.add_argument(
        '--load-results',
        type=str,
        help='Caminho para arquivo CSV com resultados existentes para gerar apenas plots'
    )
    
    args = parser.parse_args()
    
    # Executar seleção
    selector = ClassifierSelector(
        results_dir=args.output_dir,
        datasets_config=args.datasets_config,
        classifiers_config=args.classifiers_config
    )
    
    if args.load_results:
        print(f"\nCarregando resultados de: {args.load_results}")
        df = pd.read_csv(args.load_results)
    else:
        print("\nIniciando avaliação de classificadores...")
        selector.run_evaluation()
        
        print("\nSalvando resultados...")
        df, _ = selector.save_results()
    
    # Usar módulo de análise
    print_analysis_summary(df)
    generate_all_plots(df, selector.results_dir)
    
    print("\n" + "="*70)
    print("CONCLUÍDO!")
    print("="*70)
    print(f"\nResultados disponíveis em: {selector.results_dir}")


if __name__ == '__main__':
    main()
