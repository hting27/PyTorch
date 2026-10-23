"""直接讀取 archive.zip；舊溫度資料範例及註解見 legacy/dataset.py。"""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

NUMERIC = ['year', 'num_minifigs', 'num_unique_colors']
CATEGORICAL = ['theme_name', 'parent_theme']

def prepare_data(archive='archive.zip'):
    with zipfile.ZipFile(archive) as z:
        df = pd.read_csv(z.open('lego_sets_1975_2025.csv'))
    raw_count = len(df)
    df = df.drop_duplicates('set_num').copy()
    for col in NUMERIC + ['num_parts']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['year', 'num_parts'])
    df = df[(df.num_parts > 0) & df.year.between(1975, 2025)].copy()
    for col in ['num_minifigs', 'num_unique_colors']:
        df.loc[df[col] < 0, col] = np.nan
    # 以完整年份分割；同年份不跨組，模擬預測新套組。
    counts = df.groupby('year').size().sort_index().cumsum()
    train_year = int(counts.index[np.searchsorted(counts.values, len(df)*.70)])
    val_year = int(counts.index[np.searchsorted(counts.values, len(df)*.85)])
    frames = [df[df.year <= train_year], df[(df.year > train_year) & (df.year <= val_year)], df[df.year > val_year]]
    if any(len(f) == 0 for f in frames):
        raise ValueError('年份切分產生空資料集')
    train = frames[0]
    # 所有填補、標準化與類別字典只由訓練集估計。
    medians = train[NUMERIC].median().fillna(0)
    def numeric(f):
        a = f[NUMERIC].fillna(medians).to_numpy(dtype=np.float32)
        a[:, 1:] = np.log1p(a[:, 1:])
        return a
    mean, std = numeric(train).mean(0), numeric(train).std(0)
    std[std < 1e-6] = 1
    vocab = {c: sorted(train[c].fillna('Unknown').astype(str).unique().tolist()) for c in CATEGORICAL}
    def encode(f):
        blocks = [(numeric(f)-mean)/std]
        for c in CATEGORICAL:
            mapping = {v:i+1 for i,v in enumerate(vocab[c])}
            ids = f[c].fillna('Unknown').astype(str).map(mapping).fillna(0).to_numpy(dtype=int)
            blocks.append(np.eye(len(mapping)+1, dtype=np.float32)[ids])
        return np.concatenate(blocks, axis=1).astype(np.float32)
    x = [encode(f) for f in frames]
    y = [f.num_parts.to_numpy(dtype=np.float32) for f in frames]
    metadata = dict(raw_rows=raw_count, usable_rows=len(df), removed_rows=raw_count-len(df),
                    split_counts=[len(f) for f in frames], train_end_year=train_year, val_end_year=val_year,
                    input_dim=x[0].shape[1], numeric=NUMERIC, categorical=CATEGORICAL,
                    medians=medians.to_dict(), mean=mean.tolist(), std=std.tolist(), vocab=vocab)
    return x, y, frames, metadata
