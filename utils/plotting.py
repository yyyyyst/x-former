"""Publication-quality plots: 600 DPI, available serif font, colorblind-friendly."""
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


def _select_serif_font() -> str:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    return 'Times New Roman' if 'Times New Roman' in available_fonts else 'DejaVu Serif'


# Global style settings
plt.rcParams.update({
    'font.family': [_select_serif_font()],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ColorBrewer Set2 — colorblind-friendly
COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
          '#ffd92f', '#e5c494', '#b3b3b3']


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   title: str = "ROC Curve",
                   save_path: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=COLORS[0], lw=1.5, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('1 - Specificity (FPR)')
    ax.set_ylabel('Sensitivity (TPR)')
    ax.set_title(title)
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=600)
        fig.savefig(path.with_suffix('.png'), dpi=600)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = "Confusion Matrix",
                          save_path: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Good (0)', 'Poor (1)'])
    disp.plot(cmap='Blues', ax=ax, colorbar=False, values_format='d')
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=600)
        fig.savefig(path.with_suffix('.png'), dpi=600)
    plt.close(fig)


def plot_ablation_bar(results: Dict[str, Dict[str, float]],
                      metric: str = "auc",
                      title: str = "Ablation Study",
                      save_path: Optional[str] = None) -> None:
    """Bar chart comparing ablation results. results: {exp_name: {metric: val}}."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    values = [results[n][metric] for n in names]
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
    bars = ax.barh(names, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    ax.invert_yaxis()
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=9)
    fig.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=600)
        fig.savefig(path.with_suffix('.png'), dpi=600)
    plt.close(fig)
