"""直接讀 ZIP，固定分割並快取縮圖；不更動原壓縮檔。"""
import hashlib
import io
import random
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torch

CLASSES = ['freshapples', 'freshbanana', 'freshoranges',
           'rottenapples', 'rottenbanana', 'rottenoranges']


def prepare_data(archive, size=64, seed=42):
    rng = random.Random(seed)
    records, images, seen = [], [], {}
    skipped = []
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist()
                       if n.startswith(('dataset/train/', 'dataset/test/'))
                       and Path(n).suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not names:
            raise ValueError('ZIP 缺少 dataset/train 或 dataset/test 影像')
        for name in names:
            parts = name.split('/')
            if parts[2] not in CLASSES:
                raise ValueError(f'未知類別: {name}')
            raw = z.read(name)
            # 以原始檔案內容雜湊去除完全相同的影像（含跨組重複）。
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen:
                if seen[digest] != parts[2]:
                    raise ValueError('重複影像有矛盾標籤')
                continue
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    a = np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((size, size)), dtype=np.uint8).copy()
            except (OSError, ValueError) as exc:
                skipped.append({'file': name, 'reason': str(exc)})
                continue
            seen[digest] = parts[2]
            images.append(a)
            records.append({'file': name, 'label': CLASSES.index(parts[2]),
                            'split': parts[1], 'sha256': digest})
    # 保留原 test；只由 train 逐類抽 15% 作驗證。
    for label in range(len(CLASSES)):
        ids = [i for i, r in enumerate(records) if r['split'] == 'train' and r['label'] == label]
        rng.shuffle(ids)
        if len(ids) < 2:
            raise ValueError('每個類別至少需兩張訓練圖')
        for i in ids[:max(1, round(len(ids)*.15))]:
            records[i]['split'] = 'val'
    counts = {s: dict(Counter(CLASSES[r['label']] for r in records if r['split'] == s)) for s in ('train', 'val', 'test')}
    if any(len(counts[s]) != len(CLASSES) for s in counts):
        raise ValueError('分割後各組都必須包含六類')
    return np.stack(images), records, {'counts': counts, 'skipped': skipped,
        'canonical_entries': len(names), 'unique_images': len(records),
        'duplicates_removed': len(names)-len(records)-len(skipped), 'image_size': size,
        'limitation': 'Exact byte duplicates removed; near-duplicate/augmentation families are not identified. Random validation may be optimistic.'}


class FruitDataset(Dataset):
    def __init__(self, images, records, split):
        self.images, self.records, self.split = images, records, split
        self.ids = [i for i, r in enumerate(records) if r['split'] == split]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        i = self.ids[index]
        x = torch.from_numpy(self.images[i].copy()).permute(2, 0, 1).float()/255
        # 資料增強只套用訓練集；驗證與測試保持固定。
        if self.split == 'train' and torch.rand(()) < .5:
            x = x.flip(2)
        return (x-.5)/.5, self.records[i]['label']
