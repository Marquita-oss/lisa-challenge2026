"""
slice_ablation.py — Ablación del muestreo de slices en Task 1A (SOLO inferencia).

Responde a la crítica de los revisores LISA 2026 sobre la selección fija de tres
cortes (P25/P50/P75): ¿cuánta señal se pierde por muestrear el eje delgado en
posiciones fijas, y un muestreo más denso la recupera?

Reevalúa los MISMOS diez checkpoints (2 backbones x 5 folds) sobre el MISMO split
StratifiedGroupKFold (semilla 42), cambiando únicamente qué cortes forman los tres
canales de entrada. Cada imagen se predice sólo con el modelo del fold que no la vio,
así que la propiedad out-of-fold sin fugas se conserva exactamente igual que en oof.py.

IMPORTANTE: los modelos fueron ENTRENADOS con P25/P50/P75. Un muestreo distinto en
inferencia mide la robustez del muestreo, no la pregunta completa "¿y si se entrenara
denso?". Eso queda declarado como trabajo futuro en el paper.

Uso:
  python slice_ablation.py --data-root /ruta/a/data [--configs S1-central S3-pct ...]

Salida: results/slice_ablation.json  +  results/slice_ablation_probs.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG
from inference import TTA_OPS, load_model, val_transform


# ---------------------------------------------------------------------------
# Definición de los esquemas de muestreo
# ---------------------------------------------------------------------------

def triplets_for(n: int, scheme: str) -> list[tuple[int, int, int]]:
    """Índices de corte (i,j,k) que forman los 3 canales, por tripleta a promediar.

    n es el número de cortes del eje delgado. Todos los esquemas caen de vuelta a
    P25/P50/P75 si el volumen es demasiado fino para el esquema pedido.
    """
    def at(fracs):
        return tuple(min(n - 1, max(0, int(n * f))) for f in fracs)

    pct3 = [at((0.25, 0.50, 0.75))]
    if n < 3:
        return pct3

    if scheme == 'S1-central':
        m = n // 2
        return [(m, m, m)]
    if scheme == 'S3-pct':
        return pct3
    if scheme == 'S5-dense':
        p = [min(n - 1, max(0, int(n * f))) for f in (0.10, 0.30, 0.50, 0.70, 0.90)]
        return [tuple(p[i:i + 3]) for i in range(len(p) - 2)]
    if scheme == 'S9-dense':
        p = [min(n - 1, max(0, int(n * f))) for f in np.linspace(0.10, 0.90, 9)]
        return [tuple(p[i:i + 3]) for i in range(len(p) - 2)]
    if scheme == 'Sall-slide':
        if n < 5:
            return pct3
        return [(i - 1, i, i + 1) for i in range(1, n - 1)]
    raise ValueError(f"esquema desconocido: {scheme}")


def slice_to_u8(s: np.ndarray) -> np.ndarray:
    """Ventaneo p1--p99 y escalado a uint8 — idéntico a utils/io.load_lf_multichannel."""
    p1, p99 = np.percentile(s, [1, 99])
    denom = p99 - p1
    s = np.clip(s, p1, p99)
    s = (s - p1) / denom if denom > 1e-8 else np.zeros_like(s)
    return (s * 255).astype(np.uint8)


class TripletDataset(Dataset):
    """Por imagen devuelve (T,3,H,W): todas las tripletas del esquema, ya normalizadas."""

    def __init__(self, df, data_dir: Path, scheme: str, transform):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.scheme = scheme
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from utils.io import find_image_path
        row = self.df.iloc[idx]
        path = find_image_path(row['filename'], self.data_dir)
        if path is None:
            raise FileNotFoundError(row['filename'])
        vol = nib.load(str(path)).get_fdata(dtype=np.float32)

        if vol.ndim == 2:
            planes = {0: slice_to_u8(vol)}
            trips = [(0, 0, 0)]
        else:
            thin = int(np.argmin(vol.shape))
            n = vol.shape[thin]
            trips = triplets_for(n, self.scheme)
            needed = sorted({i for t in trips for i in t})
            planes = {}
            for i in needed:
                sl = [slice(None)] * 3
                sl[thin] = i
                planes[i] = slice_to_u8(vol[tuple(sl)])

        imgs = []
        for t in trips:
            arr = np.stack([planes[i] for i in t], axis=-1)
            imgs.append(self.transform(Image.fromarray(arr)))
        return torch.stack(imgs), int(row['index'])


def collate_single(batch):
    return batch[0]


@torch.no_grad()
def predict_scheme(model, df_val, data_dir, scheme, device, tta: bool, chunk: int = 64):
    """(p1,p2,idx) promediando probabilidades sobre tripletas x TTA."""
    ds = TripletDataset(df_val, data_dir, scheme, val_transform(CFG.IMG_SIZE))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=collate_single, pin_memory=True)
    ops = TTA_OPS if tta else [TTA_OPS[0]]
    out1, out2, idxs = [], [], []
    for imgs, gidx in loader:
        imgs = imgs.to(device, non_blocking=True)
        acc1, acc2 = [], []
        for op in ops:
            x = op(imgs)
            for s in range(0, x.shape[0], chunk):
                lo1, lo2 = model(x[s:s + chunk])
                acc1.append(torch.sigmoid(lo1))
                acc2.append(torch.sigmoid(lo2))
        out1.append(torch.cat(acc1).mean(0).cpu().numpy())
        out2.append(torch.cat(acc2).mean(0).cpu().numpy())
        idxs.append(gidx)
    return np.stack(out1), np.stack(out2), np.array(idxs)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def evaluate(p1, p2, gt, plane, t1, t2, cols):
    pred = np.where(p2 >= t2, 2, np.where(p1 >= t1, 1, 0))
    pos = gt >= 1
    res = {
        'micro_acc': float((pred == gt).mean()),
        'recall_pos': float((pred[pos] >= 1).mean()),
        'exact_on_pos': float((pred[pos] == gt[pos]).mean()),
        'by_plane': {},
        'by_plane_class': {},
    }
    for pl in ('axi', 'cor', 'sag'):
        m = plane == pl
        pm = gt[m] >= 1
        res['by_plane'][pl] = {
            'n_images': int(m.sum()),
            'n_pos_cells': int(pm.sum()),
            'micro_acc': float((pred[m] == gt[m]).mean()),
            'recall_pos': float((pred[m][pm] >= 1).mean()),
            'exact_on_pos': float((pred[m][pm] == gt[m][pm]).mean()),
        }
        res['by_plane_class'][pl] = {}
        for j, c in enumerate(cols):
            pj = gt[m, j] >= 1
            res['by_plane_class'][pl][c] = {
                'n_pos': int(pj.sum()),
                'recall_pos': float((pred[m, j][pj] >= 1).mean()) if pj.sum() else None,
            }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True,
                    help='carpeta data/ con metadata/ y train/')
    ap.add_argument('--configs', nargs='+',
                    default=['S1-central', 'S3-pct', 'S5-dense', 'S9-dense', 'Sall-slide'])
    ap.add_argument('--tta', choices=['on', 'off'], default='off',
                    help="TTA dihedral x8. 'off' reproduce el protocolo OOF del paper")
    ap.add_argument('--out', default='slice_ablation')
    args = ap.parse_args()

    root = Path(args.data_root)
    CFG.DATA_DIR = root
    CFG.TRAIN_DIR = root / 'train'
    CFG.CSV_PATH = root / 'metadata' / 'lisa_task1a_2026.csv'

    from data import stratified_kfold_splits

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cols = CFG.ARTIFACT_COLS
    df, splits = stratified_kfold_splits(CFG.CSV_PATH, CFG.TRAIN_DIR, CFG.N_FOLDS)
    N = len(df)
    plane = df['filename'].str.extract(r'(?i)_(axi|cor|sag)\.')[0].str.lower().values
    assert not any(p is None or p != p for p in plane), 'plano no reconocido en algún filename'
    gt = (df[cols].values >= 1).astype(int) + (df[cols].values >= 2).astype(int)

    th = json.loads((CFG.RESULTS_DIR / 'thresholds.json').read_text())
    t1 = np.array([th['t1_per_class'][c] for c in cols])
    t2 = np.array([th['t2_per_class'][c] for c in cols])

    results, store = {}, {}
    for scheme in args.configs:
        tta = args.tta == 'on'
        t0 = time.time()
        print(f"\n=== {scheme}  (TTA={'x8' if tta else 'off'}) ===", flush=True)
        agg1 = np.zeros((N, len(cols)), dtype=np.float64)
        agg2 = np.zeros((N, len(cols)), dtype=np.float64)
        for backbone, lab in CFG.BACKBONES:
            for k, (_, df_val) in enumerate(splits):
                ckpt = CFG.CHECKPOINTS_DIR / f'best_{lab}_fold{k}.pth'
                model, _ = load_model(backbone, ckpt, device, CFG.HIDDEN_DIM)
                p1, p2, idx = predict_scheme(model, df_val, CFG.TRAIN_DIR, scheme, device, tta)
                agg1[idx] += p1 / len(CFG.BACKBONES)
                agg2[idx] += p2 / len(CFG.BACKBONES)
                del model
                torch.cuda.empty_cache()
                print(f"  {lab} fold{k}: {len(df_val)} imgs  ({time.time()-t0:.0f}s)", flush=True)

        n_trip = len(triplets_for(36, scheme))
        results[scheme] = {
            'tta': 'x8' if tta else 'off',
            'triplets_at_n36': n_trip,
            'calibrated': evaluate(agg1, agg2, gt, plane, t1, t2, cols),
            'at_0.5': evaluate(agg1, agg2, gt, plane, 0.5, 0.5, cols),
            'seconds': round(time.time() - t0, 1),
        }
        store[f'{scheme}_p1'] = agg1.astype(np.float32)
        store[f'{scheme}_p2'] = agg2.astype(np.float32)
        c = results[scheme]['calibrated']
        print(f"  -> micro-acc {c['micro_acc']:.4f} | recall+ {c['recall_pos']:.3f} | "
              f"axi/cor/sag {c['by_plane']['axi']['recall_pos']:.3f}/"
              f"{c['by_plane']['cor']['recall_pos']:.3f}/"
              f"{c['by_plane']['sag']['recall_pos']:.3f}", flush=True)

        (CFG.RESULTS_DIR / f'{args.out}.json').write_text(json.dumps(results, indent=2))
        np.savez_compressed(CFG.RESULTS_DIR / f'{args.out}_probs.npz', gt=gt, plane=plane, **store)

    print("\nListo:", CFG.RESULTS_DIR / f'{args.out}.json')


if __name__ == '__main__':
    main()
