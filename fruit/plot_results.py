"""從已保存的結果重建圖表，不必重新訓練。"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(out):
    out = Path(out)
    r = json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    with (out/'loss.csv').open(encoding='utf-8') as f:
        rows = list(csv.reader(f))[1:]
    h = np.asarray(rows, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, cols, title in zip(axes, ((1, 2), (3, 4)), ('Cross entropy', 'Accuracy')):
        for col, label in zip(cols, ('train', 'validation')):
            ax.plot(h[:, 0], h[:, col], label=label)
        ax.set(xlabel='Epoch', title=title); ax.legend()
    fig.tight_layout(); fig.savefig(out/'learning_curves.png', dpi=160); plt.close(fig)
    cm = np.asarray(r['confusion_matrix'])
    classes = [x['class'] for x in r['per_class']]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap='Blues')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='Actual', title='Test confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    for i in range(6):
        for j in range(6): ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
    fig.tight_layout(); fig.savefig(out/'confusion_matrix.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    plot_results(Path(__file__).resolve().parent/'result')
