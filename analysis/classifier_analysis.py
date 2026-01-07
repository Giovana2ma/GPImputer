"""
Análise e visualização de resultados de classificadores.

Este módulo fornece funções para:
- Gerar visualizações de comparação de classificadores
- Analisar tradeoff entre F1-score e tempo de execução
- Criar relatórios de performance
"""

from pathlib import Path
from datetime import datetime
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

sns.set_style("whitegrid")


def plot_f1_heatmap(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Gera heatmap de F1-score por classificador e dataset.
    
    Args:
        df: DataFrame com resultados
        output_dir: Diretório para salvar a visualização
        
    Returns:
        Nome do arquivo salvo
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_f1 = df.pivot(index='classifier', columns='dataset', values='f1_score_mean')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax)
    ax.set_title('F1-Score por Classificador e Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'f1_heatmap_{timestamp}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def plot_execution_time(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Gera gráfico de tempo de execução por classificador.
    
    Args:
        df: DataFrame com resultados
        output_dir: Diretório para salvar a visualização
        
    Returns:
        Nome do arquivo salvo
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    time_summary = df.groupby('classifier')['execution_time'].mean().sort_values()
    time_summary.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Tempo Médio de Execução (s)', fontsize=12)
    ax.set_ylabel('Classificador', fontsize=12)
    ax.set_title('Tempo de Execução por Classificador', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    filename = f'execution_time_{timestamp}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def plot_tradeoff_scatter(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Gera scatter plot de F1-score vs tempo (análise de tradeoff).
    
    Args:
        df: DataFrame com resultados
        output_dir: Diretório para salvar a visualização
        
    Returns:
        Nome do arquivo salvo
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuração para artigo científico
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 15,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 13,
        'figure.titlesize': 18
    })
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    summary = df.groupby('classifier').agg({
        'f1_score_mean': 'mean',
        'execution_time': 'mean'
    }).reset_index()
    
    # Definir marcadores para diferenciar classificadores
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    if len(summary) > len(markers):
        markers = markers * (len(summary) // len(markers) + 1)
    
    # Plotar com seaborn para facilitar markers e cores
    sns.scatterplot(
        data=summary,
        x='f1_score_mean',
        y='execution_time',
        hue='classifier',
        style='classifier',
        markers=markers[:len(summary)],
        s=250,
        alpha=0.9,
        palette='viridis',
        edgecolor='k',
        linewidth=1.5,
        ax=ax,
        zorder=3
    )
    
    ax.set_xlabel('F1-Score Médio')
    ax.set_ylabel('Tempo Médio de Execução (s)')
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    # Linhas de referência (Top 25%)
    f1_threshold = summary['f1_score_mean'].quantile(0.75)
    time_threshold = summary['execution_time'].quantile(0.25)
    
    ax.axvline(f1_threshold, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax.axhline(time_threshold, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
    # Anotação da região ideal
    

    # Legenda externa na parte inferior
    ax.legend(
        bbox_to_anchor=(0.5, -0.15), 
        loc='upper center', 
        ncol=3, 
        frameon=False,
        columnspacing=3.0,  # Aumenta distância entre colunas
        handletextpad=0.5,  # Ajusta distância entre marcador e texto
        labelspacing=1.5    # Aumenta distância vertical entre linhas
    )
    
    plt.tight_layout()
    
    filename = f'tradeoff_scatter_{timestamp}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def plot_f1_boxplot(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Gera box plot de F1-score por classificador.
    
    Args:
        df: DataFrame com resultados
        output_dir: Diretório para salvar a visualização
        
    Returns:
        Nome do arquivo salvo
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    df_sorted = df.sort_values('f1_score_mean', ascending=False)
    classifiers_order = df_sorted.groupby('classifier')['f1_score_mean'].mean().sort_values(ascending=False).index
    
    sns.boxplot(data=df, x='classifier', y='f1_score_mean', 
               order=classifiers_order, ax=ax, palette='Set3')
    ax.set_xlabel('Classificador', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Distribuição de F1-Score por Classificador', 
                fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filename = f'f1_boxplot_{timestamp}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def plot_performance_by_dataset(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Gera gráficos de performance separados por dataset.
    
    Args:
        df: DataFrame com resultados
        output_dir: Diretório para salvar a visualização
        
    Returns:
        Nome do arquivo salvo
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = (n_datasets + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_datasets > 1 else [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = df[df['dataset'] == dataset].sort_values('f1_score_mean', ascending=False)
        
        ax.barh(range(len(data)), data['f1_score_mean'], 
               color=plt.cm.viridis(np.linspace(0, 1, len(data))))
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['classifier'], fontsize=9)
        ax.set_xlabel('F1-Score', fontsize=10)
        ax.set_title(f'Dataset: {dataset}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Adicionar valores
        for i, (_, row) in enumerate(data.iterrows()):
            ax.text(row['f1_score_mean'], i, f" {row['f1_score_mean']:.3f}",
                   va='center', fontsize=8)
    
    # Remover eixos extras
    for idx in range(len(datasets), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    filename = f'performance_by_dataset_{timestamp}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def generate_all_plots(df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    """
    Gera todas as visualizações de análise.
    
    Args:
        df: DataFrame com resultados
        output_dir: Diretório para salvar as visualizações
        
    Returns:
        Dicionário com nome do plot e arquivo salvo
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remover XGBoost dos plots
    df = df[~df['classifier'].astype(str).str.contains('XGBoost|xgboost', case=False, regex=True)]
    
    plots = {}
    
    print("\nGerando visualizações...")
    
    plots['heatmap'] = plot_f1_heatmap(df, output_dir)
    print(f"  ✓ {plots['heatmap']}")
    
    plots['execution_time'] = plot_execution_time(df, output_dir)
    print(f"  ✓ {plots['execution_time']}")
    
    plots['tradeoff'] = plot_tradeoff_scatter(df, output_dir)
    print(f"  ✓ {plots['tradeoff']}")
    
    plots['boxplot'] = plot_f1_boxplot(df, output_dir)
    print(f"  ✓ {plots['boxplot']}")
    
    plots['by_dataset'] = plot_performance_by_dataset(df, output_dir)
    print(f"  ✓ {plots['by_dataset']}")
    
    return plots


def analyze_tradeoff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analisa tradeoff entre F1-score e tempo de execução.
    
    Args:
        df: DataFrame com resultados
        
    Returns:
        DataFrame com análise de tradeoff ordenado
    """
    # Agregar por classificador
    summary = df.groupby('classifier').agg({
        'f1_score_mean': 'mean',
        'f1_score_std': 'mean',
        'execution_time': 'mean'
    })
    
    # Normalizar métricas para calcular tradeoff
    f1_norm = (summary['f1_score_mean'] - summary['f1_score_mean'].min()) / \
              (summary['f1_score_mean'].max() - summary['f1_score_mean'].min())
    
    time_norm = (summary['execution_time'] - summary['execution_time'].min()) / \
                (summary['execution_time'].max() - summary['execution_time'].min())
    
    # Score de tradeoff: maximizar F1, minimizar tempo
    # Peso maior para F1 (0.7) vs tempo (0.3)
    summary['tradeoff_score'] = f1_norm * 0.7 - time_norm * 0.3
    
    # Ordenar por tradeoff
    summary = summary.sort_values('tradeoff_score', ascending=False)
    
    return summary


def print_analysis_summary(df: pd.DataFrame):
    """
    Imprime resumo da análise no terminal.
    
    Args:
        df: DataFrame com resultados
    """
    print("\n" + "="*70)
    print("RESUMO DA ANÁLISE")
    print("="*70)
    
    # Ranking por F1-score
    print("\nRANKING POR F1-SCORE:")
    ranking = df.groupby('classifier')['f1_score_mean'].mean().sort_values(ascending=False)
    for i, (clf, score) in enumerate(ranking.head(5).items(), 1):
        print(f"  {i}. {clf}: {score:.4f}")
    
    # Ranking por velocidade
    print("\nRANKING POR VELOCIDADE (menor tempo):")
    speed_ranking = df.groupby('classifier')['execution_time'].mean().sort_values()
    for i, (clf, time) in enumerate(speed_ranking.head(5).items(), 1):
        print(f"  {i}. {clf}: {time:.2f}s")
    
    # Análise de tradeoff
    print("\nANÁLISE DE TRADEOFF (F1 vs Tempo):")
    tradeoff = analyze_tradeoff(df)
    for i, (clf, row) in enumerate(tradeoff.head(5).iterrows(), 1):
        print(f"  {i}. {clf}: F1={row['f1_score_mean']:.4f}, "
              f"Tempo={row['execution_time']:.2f}s, "
              f"Score={row['tradeoff_score']:.4f}")
    
    print("\n" + "="*70)
