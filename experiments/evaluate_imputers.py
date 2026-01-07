#!/usr/bin/env python3
"""
Avalia imputadores (mean e median) usando LogisticRegression e calcula F1-score.

Uso:
  python3 experiments/evaluate_imputers.py \
    --config config/datasets_config.yaml \
    --output results/imputer_evaluation.csv \
    [--inject-missingness 0.1]

O script carrega os datasets configurados em `datasets_config.yaml`, aplica
one-hot em categóricas (preenchendo missings com um placeholder), opcionalmente
injeta missingness aleatória, e avalia dois imputadores ('mean' e 'median')
usando StratifiedKFold e `LogisticRegression`.
"""
from pathlib import Path
import argparse
import csv
import time
import traceback
import sys
import os

# Add project root to sys.path to allow importing 'data' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

from data.dataset_loader import load_dataset_configs

try:
    from ucimlrepo import fetch_ucirepo
except Exception as e:
    fetch_ucirepo = None


def safe_onehot(X_cat: pd.DataFrame):
    if X_cat.shape[1] == 0:
        return None, []

    # Preencher NaNs com placeholder para one-hot
    X_cat_filled = X_cat.fillna("__MISSING__")
    encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    X_cat_enc = encoder.fit_transform(X_cat_filled)
    feature_names = list(encoder.get_feature_names_out(X_cat.columns))
    return X_cat_enc, feature_names


def evaluate_on_dataset(dataset_name, cfg, inject_missingness, n_splits):
    if fetch_ucirepo is None:
        raise ImportError("ucimlrepo não está disponível. Instale com: pip install ucimlrepo")

    print(f"\nCarregando dataset {dataset_name} (UCI id: {cfg['id']})...")
    ds = fetch_ucirepo(id=cfg['id'])
    X_df = ds.data.features.copy()
    y_raw = ds.data.targets.values.ravel()

    # Aplicar binarização de forma robusta (evitar astype(int) sobre NaNs)
    try:
        y = cfg['binarize_lambda'](y_raw)
    except Exception:
        btype = cfg.get('binarize_type', 'equal')
        bval = cfg.get('binarize_value')
        if btype == 'equal':
            y = np.where(np.equal(y_raw, bval), 1, 0)
        elif btype == 'greater':
            # tenta comparar numericamente; valores não numéricos produzirão False
            try:
                y = np.where(np.greater(y_raw.astype(float), float(bval)), 1, 0)
            except Exception:
                y = np.where(np.greater(y_raw, bval), 1, 0)
        else:
            y = np.where(np.equal(y_raw, bval), 1, 0)

    # Separar numéricas e categóricas
    X_num = X_df.select_dtypes(include=[np.number])
    X_cat = X_df.select_dtypes(exclude=[np.number])

    # One-hot nas categóricas (mantendo NaNs preenchidos por placeholder)
    X_cat_enc, cat_feature_names = safe_onehot(X_cat)

    # Combinar arrays (numéricas podem conter NaNs)
    if X_cat_enc is not None:
        if X_num.shape[1] > 0:
            X_full = np.hstack([X_num.values, X_cat_enc])
        else:
            X_full = X_cat_enc
    else:
        X_full = X_num.values

    # Garantir dtype float para que possamos atribuir NaN ao injetar missingness
    try:
        X_full = X_full.astype(float)
    except Exception:
        X_full = np.array(X_full, dtype=float)

    # Opcional: injetar missingness aleatória no dataset (apenas em features numéricas/encoded)
    rng = np.random.default_rng(42)
    if inject_missingness > 0:
        n_entries = X_full.size
        n_missing = int(np.floor(inject_missingness * n_entries))
        idx = rng.choice(n_entries, size=n_missing, replace=False)
        flat = X_full.ravel()
        flat[idx] = np.nan
        X_full = flat.reshape(X_full.shape)

    # Filtrar amostras com target NaN ou indefinido
    mask = ~pd.isna(y)
    X_full = X_full[mask]
    y = y[mask]

    # Ajustar n_splits se necessário (cada classe precisa ter >= n_splits exemplos)
    classes, counts = np.unique(y, return_counts=True)
    min_class_count = counts.min()
    if min_class_count < 2:
        print(f"  → Pulando {dataset_name}: classe com menos de 2 exemplos")
        return []

    folds = min(n_splits, min_class_count)
    if folds < 2:
        print(f"  → Pulando {dataset_name}: folds insuficientes ({folds})")
        return []

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    results = []

    for strategy in ("mean", "median"):
        scores = []
        for train_idx, test_idx in skf.split(X_full, y):
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            imputer = SimpleImputer(strategy=strategy)
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train_imp, y_train)
            y_pred = clf.predict(X_test_imp)
            try:
                sc = f1_score(y_test, y_pred)
            except Exception:
                sc = float('nan')
            scores.append(sc)

        mean_f1 = float(np.nanmean(scores))
        std_f1 = float(np.nanstd(scores))
        results.append({
            'dataset': dataset_name,
            'strategy': strategy,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'n_samples': int(X_full.shape[0]),
            'n_features': int(X_full.shape[1]),
            'n_folds': folds
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/datasets_config.yaml')
    parser.add_argument('--output', default='results/imputer_evaluation.csv')
    parser.add_argument('--inject-missingness', type=float, default=0.0,
                        help='Fração de entradas a tornar NaN para testar imputação (0-1)')
    parser.add_argument('--n-splits', type=int, default=5)
    args = parser.parse_args()

    cfgs = load_dataset_configs(args.config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    start = time.time()
    for name, cfg in cfgs.items():
        try:
            res = evaluate_on_dataset(name, cfg, args.inject_missingness, args.n_splits)
            all_results.extend(res)
        except Exception as e:
            print(f"Erro avaliando {name}: {e}")
            print(traceback.format_exc())

    if len(all_results) == 0:
        print("Nenhum resultado gerado.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(out_path, index=False)
    elapsed = time.time() - start
    print(f"\nResultados salvos em: {out_path} (tempo: {elapsed:.1f}s)")


if __name__ == '__main__':
    main()
