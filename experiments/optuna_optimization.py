"""
Otimização de hiperparâmetros de imputadores usando Optuna.

Este script otimiza os hiperparâmetros dos imputadores (KNN, MICE, MissForest, SVD)
usando Optuna, avaliando a qualidade da imputação através do F1-score de um classificador.
"""

import os
import sys
import yaml
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from imputers.knn_imputer import KNNImputerWrapper
from imputers.mice_imputer import MICEImputerWrapper
from imputers.missforest_imputer import MissForestImputerWrapper
from imputers.svd_imputer import SVDImputerWrapper
from data.missingness import generate_missing_data
from data.dataset_loader import load_dataset_configs, load_and_preprocess_dataset

warnings.filterwarnings('ignore')


class OptunaImputer:
    """Classe para otimização de imputadores com Optuna."""
    
    def __init__(self, config_path: str, datasets_config_path: str = "config/datasets_config.yaml"):
        """
        Inicializa o otimizador.
        
        Args:
            config_path: Caminho para o arquivo de configuração YAML do Optuna
            datasets_config_path: Caminho para o arquivo de configuração dos datasets
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config['results']['output_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Carregar configurações de datasets
        self.dataset_configs = load_dataset_configs(datasets_config_path)
        
        self.all_results = []
        
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega um dataset usando o dataset_loader.
        
        Args:
            dataset_name: Nome do dataset
            
        Returns:
            X, y: Features e target
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset {dataset_name} não encontrado em datasets_config.yaml")
        
        config = self.dataset_configs[dataset_name]
        X, y = load_and_preprocess_dataset(dataset_name, config, verbose=False)
        
        return X, y
    
    def create_classifier(self) -> Any:
        """Cria o classificador baseado na configuração."""
        clf_config = self.config['classifier']
        clf_type = clf_config['type']
        params = clf_config.get('params', {})
        
        # Mapeamento de tipos para classes
        classifier_map = {
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'decision_tree': DecisionTreeClassifier,
            'knn': KNeighborsClassifier
        }
        
        if clf_type not in classifier_map:
            raise ValueError(f"Unknown classifier type: {clf_type}")
        
        return classifier_map[clf_type](**params)
    
    def suggest_params(self, trial: optuna.Trial, imputer_name: str) -> Dict[str, Any]:
        """
        Sugere hiperparâmetros para um trial usando Optuna.
        
        Args:
            trial: Trial do Optuna
            imputer_name: Nome do imputador
            
        Returns:
            Dicionário com os parâmetros sugeridos
        """
        params_config = self.config['imputers'][imputer_name]['params']
        params = {}
        
        for param_name, param_spec in params_config.items():
            param_type = param_spec['type']
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_spec['low'],
                    param_spec['high'],
                    step=param_spec.get('step', 1)
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_spec['low'],
                    param_spec['high']
                )
            elif param_type == 'loguniform':
                params[param_name] = trial.suggest_float(
                    param_name,
                    float(param_spec['low']),
                    float(param_spec['high']),
                    log=True
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_spec['choices']
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        
        return params
    
    def create_imputer(self, imputer_name: str, params: Dict[str, Any]) -> Any:
        """
        Cria um imputador com os parâmetros especificados.
        
        Args:
            imputer_name: Nome do imputador
            params: Parâmetros do imputador
            
        Returns:
            Instância do imputador
        """
        random_state = self.config['random_state']
        
        if imputer_name == 'knn':
            return KNNImputerWrapper(**params)
        
        elif imputer_name == 'mice':
            return MICEImputerWrapper(**params, random_state=random_state)
        
        elif imputer_name == 'missforest':
            return MissForestImputerWrapper(**params, random_state=random_state)
        
        elif imputer_name == 'svd':
            return SVDImputerWrapper(**params)
        
        else:
            raise ValueError(f"Unknown imputer: {imputer_name}")
    
    def objective(
        self,
        trial: optuna.Trial,
        imputer_name: str,
        X_missing: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Função objetivo para otimização.
        
        Args:
            trial: Trial do Optuna
            imputer_name: Nome do imputador
            X_missing: Features com missing values
            y: Target
            
        Returns:
            Score médio do cross-validation
        """
        # Sugerir parâmetros
        params = self.suggest_params(trial, imputer_name)
        
        # Criar imputador
        try:
            imputer = self.create_imputer(imputer_name, params)
        except Exception as e:
            print(f"Error creating imputer: {e}")
            return 0.0
        
        # Imputar
        try:
            X_imputed = imputer.fit_transform(X_missing)
        except Exception as e:
            print(f"Error during imputation: {e}")
            return 0.0
        
        # Criar classificador
        classifier = self.create_classifier()
        
        # Cross-validation
        cv_config = self.config['cross_validation']
        cv = StratifiedKFold(
            n_splits=cv_config['n_splits'],
            shuffle=cv_config['shuffle'],
            random_state=cv_config['random_state']
        )
        
        metric = self.config['classifier']['metric']
        
        try:
            scores = cross_val_score(
                classifier,
                X_imputed,
                y,
                cv=cv,
                scoring=metric,
                n_jobs=1  # Evitar conflitos com paralelização do Optuna
            )
            return scores.mean()
        except Exception as e:
            print(f"Error during cross-validation: {e}")
            return 0.0
    
    def optimize_imputer(
        self,
        imputer_name: str,
        dataset_name: str,
        X_missing: np.ndarray,
        y: np.ndarray
    ) -> optuna.Study:
        """
        Otimiza um imputador para um dataset específico.
        
        Args:
            imputer_name: Nome do imputador
            dataset_name: Nome do dataset
            X_missing: Features com missing values
            y: Target
            
        Returns:
            Estudo do Optuna
        """
        print(f"\n{'='*70}")
        print(f"Otimizando {imputer_name.upper()} no dataset {dataset_name}")
        print(f"{'='*70}")
        
        optuna_config = self.config['optuna']
        
        # Criar estudo
        study_name = f"{optuna_config['study_name']}_{imputer_name}_{dataset_name}"
        
        pruner = MedianPruner() if optuna_config['pruning']['enabled'] else None
        
        study = optuna.create_study(
            study_name=study_name,
            direction=optuna_config['direction'],
            pruner=pruner,
            storage=optuna_config.get('storage'),
            load_if_exists=True
        )
        
        # Otimizar
        study.optimize(
            lambda trial: self.objective(trial, imputer_name, X_missing, y),
            n_trials=optuna_config['n_trials'],
            timeout=optuna_config.get('timeout'),
            n_jobs=optuna_config.get('n_jobs', 1),
            show_progress_bar=True
        )
        
        # Resultados
        print(f"\nMelhor F1-score: {study.best_value:.4f}")
        print(f"Melhores parâmetros:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        
        return study
    
    def save_results(
        self,
        study: optuna.Study,
        imputer_name: str,
        dataset_name: str,
        missing_ratio: float
    ):
        """
        Salva os resultados da otimização.
        
        Args:
            study: Estudo do Optuna
            imputer_name: Nome do imputador
            dataset_name: Nome do dataset
            missing_ratio: Percentual de missing values
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Preparar dados
        trials_data = []
        for trial in study.trials:
            trial_dict = {
                'trial_number': trial.number,
                'f1_score': trial.value,
                'imputer': imputer_name,
                'dataset': dataset_name,
                'missing_ratio': missing_ratio,
                **trial.params
            }
            trials_data.append(trial_dict)
            self.all_results.append(trial_dict)
        
        # Salvar trials individuais
        if self.config['results']['save']['all_trials']:
            df_trials = pd.DataFrame(trials_data)
            
            save_format = self.config['results']['save_format']
            
            if save_format in ['csv', 'both']:
                csv_path = self.results_dir / f"trials_{imputer_name}_{dataset_name}_miss{int(missing_ratio*100)}_{timestamp}.csv"
                df_trials.to_csv(csv_path, index=False)
                print(f"Trials salvos em: {csv_path}")
            
            if save_format in ['json', 'both']:
                json_path = self.results_dir / f"trials_{imputer_name}_{dataset_name}_miss{int(missing_ratio*100)}_{timestamp}.json"
                df_trials.to_json(json_path, orient='records', indent=2)
        
        # Salvar melhores parâmetros
        if self.config['results']['save']['best_params']:
            best_params = {
                'imputer': imputer_name,
                'dataset': dataset_name,
                'missing_ratio': missing_ratio,
                'best_f1_score': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials),
                'timestamp': timestamp
            }
            
            best_path = self.results_dir / f"best_params_{imputer_name}_{dataset_name}_miss{int(missing_ratio*100)}_{timestamp}.yaml"
            with open(best_path, 'w') as f:
                yaml.dump(best_params, f, default_flow_style=False)
        
        # Salvar objeto do estudo
        if self.config['results']['save']['study_object']:
            study_path = self.results_dir / f"study_{imputer_name}_{dataset_name}_miss{int(missing_ratio*100)}_{timestamp}.pkl"
            with open(study_path, 'wb') as f:
                pickle.dump(study, f)
        
        # Salvar plots
        if self.config['results']['save']['plots']:
            try:
                import plotly
                
                # Optimization history
                fig = optuna.visualization.plot_optimization_history(study)
                fig.write_html(self.results_dir / f"optimization_history_{imputer_name}_{dataset_name}_{timestamp}.html")
                
                # Parameter importance
                if len(study.trials) > 10:
                    fig = optuna.visualization.plot_param_importances(study)
                    fig.write_html(self.results_dir / f"param_importance_{imputer_name}_{dataset_name}_{timestamp}.html")
            except Exception as e:
                print(f"Warning: Could not save plots: {e}")
    
    def run(self):
        """Executa a otimização completa."""
        print(f"\n{'#'*70}")
        print("OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA")
        print(f"{'#'*70}")
        
        # Iterar sobre datasets configurados
        dataset_names = self.config.get('datasets', [])
        print(f"\nDatasets configurados para otimização: {dataset_names}")
        
        if not dataset_names:
            print("AVISO: Nenhum dataset configurado em optuna_config.yaml")
            return
        
        for dataset_name in dataset_names:
            print(f"\n{'='*70}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*70}")
            
            # Carregar dataset usando dataset_loader
            try:
                X, y = self.load_dataset(dataset_name)
            except Exception as e:
                print(f"ERRO ao carregar dataset {dataset_name}: {e}")
                continue
            
            # Iterar sobre ratios de missing values
            for missing_ratio in self.config['missingness']['ratios']:
                print(f"\nMissing ratio: {missing_ratio*100}%")
                
                # Introduzir missing values
                X_missing, _ = generate_missing_data(
                    X.copy(),
                    mechanism=self.config['missingness']['mechanism'],
                    missing_rate=missing_ratio,
                    seed=self.config['random_state']
                )
                
                # Iterar sobre imputadores
                for imputer_name, imputer_config in self.config['imputers'].items():
                    if not imputer_config['enabled']:
                        continue
                    
                    # Otimizar
                    study = self.optimize_imputer(
                        imputer_name,
                        dataset_name,
                        X_missing,
                        y
                    )
                    
                    # Salvar resultados
                    self.save_results(
                        study,
                        imputer_name,
                        dataset_name,
                        missing_ratio
                    )
        
        # Salvar todos os resultados consolidados
        if self.all_results:
            df_all = pd.DataFrame(self.all_results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            all_results_path = self.results_dir / f"all_optimization_results_{timestamp}.csv"
            df_all.to_csv(all_results_path, index=False)
            print(f"\n{'='*70}")
            print(f"Todos os resultados salvos em: {all_results_path}")
            print(f"{'='*70}")
            
            # Resumo
            print("\nRESUMO DA OTIMIZAÇÃO:")
            print(df_all.groupby(['imputer', 'dataset', 'missing_ratio'])['f1_score'].agg(['mean', 'max', 'count']))
            
            # Chamar análise automática se disponível
            try:
                from analysis.analyze_optuna_results import OptunaResultsAnalyzer
                print("\n" + "="*70)
                print("GERANDO ANÁLISE E VISUALIZAÇÕES...")
                print("="*70)
                analyzer = OptunaResultsAnalyzer(str(all_results_path))
                analyzer.analyze_all()
                print("\nAnálise completa!")
            except ImportError as e:
                print(f"\nWarning: Não foi possível importar analyze_optuna_results: {e}")
                print("Execute manualmente: python analysis/analyze_optuna_results.py <arquivo_csv>")


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Otimização de imputadores com Optuna')
    parser.add_argument(
        '--config',
        type=str,
        default='config/optuna_config.yaml',
        help='Caminho para o arquivo de configuração'
    )
    
    args = parser.parse_args()
    
    # Executar otimização
    optimizer = OptunaImputer(args.config)
    optimizer.run()


if __name__ == '__main__':
    main()
