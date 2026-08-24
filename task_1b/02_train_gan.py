"""Task 1b — GAN de mejora perceptual: LF -> "look de alto campo" (CISO distrib).

Enfoque NO pareado (evita el problema de registracion que hundio LF2CISO):
  - Discriminador (PatchGAN) distingue slices CISO reales (reescalados a la escala
    LF) de la salida del generador. Empuja la salida hacia la DISTRIBUCION CISO
    -> baja FID/LPIPS/FRD (las metricas dominantes).
  - Generador = ResUNet (init desde v2), residual, canvas nativo 160.
  - Anclaje L1+SSIM al LF de entrada: preserva la anatomia, evita alucinacion/drift.
  - LSGAN (estable).

Seleccion de checkpoint por FID LOCAL (piq) en val/complete -> solo nos quedamos
con el que de verdad mejora la metrica dominante. Si ninguno baja del baseline,
no se usa.

Uso:
  python task_1b/02_train_gan.py --epochs 40 --anchor-l1 2.0 --anchor-ssim 0.5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize as skr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_1b.config import TRAIN_DIR, CHECKPOINTS_DIR, RESULTS_DIR, RANDOM_SEED, VAL_COMPLETE_DIR
from task_1b.model import build_model, count_params
from task_1b.utils.metrics import ssim_torch
import piq
from piq import FID
from piq.feature_extractors import InceptionV3

CANVAS = 160
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def npct(s):
    p1, p99 = np.percentile(s, [1, 99]); d = p99 - p1
    return np.clip((s - p1) / (d if d > 1e-8 else 1), 0, 1).astype(np.float32)


def pad_to(a, size=CANVAS):
    H, W = a.shape
    out = np.zeros((size, size), np.float32); out[:H, :W] = a; return out


def build_banks(case_dirs, min_fg=0.03):
    """Devuelve (lf_slices, ciso_slices) NO pareados, nativos float16, y mask sizes."""
    lf_bank, ci_bank = [], []
    for cdir in case_dirs:
        cid = cdir.name
        ciso_p = cdir / f'lisa_{cid}_ciso.nii.gz'
        if not ciso_p.exists():
            continue
        ciso = npct(nib.load(str(ciso_p)).get_fdata(dtype=np.float32))
        for plane in ['axi', 'cor', 'sag']:
            lf_p = cdir / f'lisa_{cid}_lf_{plane}.nii.gz'
            if not lf_p.exists():
                continue
            lf_vol = nib.load(str(lf_p)).get_fdata(dtype=np.float32)
            if lf_vol.ndim != 3:
                continue
            thin = int(np.argmin(lf_vol.shape))
            ciso_rs = skr(ciso, lf_vol.shape, order=1, preserve_range=True,
                          anti_aliasing=True).astype(np.float32)
            for i in range(lf_vol.shape[thin]):
                idx = [slice(None)] * 3; idx[thin] = i
                lf_s = lf_vol[tuple(idx)]
                if lf_s.max() <= 1e-6:
                    continue
                lf_n = npct(lf_s)
                if float((lf_n > 0.1).mean()) < min_fg:
                    continue
                lf_bank.append((lf_n.astype(np.float16), lf_n.shape))
                ci_bank.append(npct(ciso_rs[tuple(idx)]).astype(np.float16))
    return lf_bank, ci_bank


class LFSet(Dataset):
    def __init__(self, lf_bank, augment=True):
        self.b = lf_bank; self.aug = augment

    def __len__(self): return len(self.b)

    def __getitem__(self, i):
        a, (H, W) = self.b[i]; a = a.astype(np.float32)
        if self.aug:
            if np.random.rand() < 0.5: a = np.fliplr(a)
            if np.random.rand() < 0.5: a = np.flipud(a)
        m = np.zeros((CANVAS, CANVAS), np.float32); m[:a.shape[0], :a.shape[1]] = 1.0
        return torch.from_numpy(pad_to(np.ascontiguousarray(a)))[None], torch.from_numpy(m)[None]


class CISet(Dataset):
    def __init__(self, ci_bank, augment=True):
        self.b = ci_bank; self.aug = augment

    def __len__(self): return len(self.b)

    def __getitem__(self, i):
        a = self.b[i].astype(np.float32)
        if self.aug:
            if np.random.rand() < 0.5: a = np.fliplr(a)
            if np.random.rand() < 0.5: a = np.flipud(a)
        return torch.from_numpy(pad_to(np.ascontiguousarray(a)))[None]


class PatchD(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        def blk(i, o, s): return nn.Sequential(
            nn.Conv2d(i, o, 4, s, 1), nn.InstanceNorm2d(o, affine=True),
            nn.LeakyReLU(0.2, True))
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2, True),   # 80
            blk(ch, ch * 2, 2),                                   # 40
            blk(ch * 2, ch * 4, 2),                               # 20
            blk(ch * 4, ch * 4, 1),                               # 20
            nn.Conv2d(ch * 4, 1, 4, 1, 1))                        # patch map
    def forward(self, x): return self.net(x)


# ── seleccion de checkpoint por FID local ──────────────────────────────────────
_INC = None
def _feats(slices, batch=64):
    global _INC
    if _INC is None: _INC = InceptionV3().to(DEV).eval()
    fs = []
    with torch.no_grad():
        for i in range(0, len(slices), batch):
            ch = slices[i:i + batch]
            t = torch.stack([torch.from_numpy(skr(s, (128, 128), order=1,
                             preserve_range=True).astype(np.float32)) for s in ch])
            t = t[:, None].repeat(1, 3, 1, 1).to(DEV)
            fs.append(_INC(t)[0].squeeze(-1).squeeze(-1).cpu())
    return torch.cat(fs)


def _slices(vol):
    thin = int(np.argmin(vol.shape)); out = []
    for i in range(vol.shape[thin]):
        idx = [slice(None)] * 3; idx[thin] = i
        s = vol[tuple(idx)]
        if s.max() > 1e-6: out.append(npct(s))
    return out


@torch.no_grad()
def local_fid(model, ref_feats):
    """Corre el generador en val/complete LF y mide FID local vs CISO."""
    model.eval()
    pred_sl = []
    for cdir in sorted(VAL_COMPLETE_DIR.iterdir()):
        if not cdir.is_dir(): continue
        for plane in ['axi', 'cor', 'sag']:
            lf_p = cdir / f'lisa_validation_{cdir.name}_lf_{plane}.nii.gz'
            if not lf_p.exists(): continue
            vol = nib.load(str(lf_p)).get_fdata(dtype=np.float32)
            thin = int(np.argmin(vol.shape))
            for i in range(vol.shape[thin]):
                idx = [slice(None)] * 3; idx[thin] = i
                s = vol[tuple(idx)]
                if s.max() <= 1e-6: continue
                sn = npct(s); H, W = sn.shape
                x = torch.from_numpy(pad_to(sn))[None, None].to(DEV)
                o = model(x)[0, 0, :H, :W].cpu().numpy()
                pred_sl.append(np.clip(o, 0, 1).astype(np.float32))
    pf = _feats(pred_sl)
    return float(FID()(pf, ref_feats).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--anchor-l1', type=float, default=2.0)
    ap.add_argument('--anchor-ssim', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--eval-every', type=int, default=2)
    args = ap.parse_args()

    np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    cases = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    print(f"Casos train: {len(cases)}  | construyendo banks...")
    t0 = time.time()
    lf_bank, ci_bank = build_banks(cases)
    print(f"  LF slices {len(lf_bank)}  CISO slices {len(ci_bank)}  ({time.time()-t0:.0f}s)")

    lf_loader = DataLoader(LFSet(lf_bank), batch_size=args.batch, shuffle=True,
                           num_workers=0, drop_last=True)
    ci_loader = DataLoader(CISet(ci_bank), batch_size=args.batch, shuffle=True,
                           num_workers=0, drop_last=True)

    G = build_model(DEV)
    v2 = CHECKPOINTS_DIR / 'best_1b_v2.pth'
    if v2.exists():
        G.load_state_dict(torch.load(v2, map_location=DEV, weights_only=False)['model_state'])
        print("  G init desde v2")
    D = PatchD().to(DEV)
    print(f"  G params {count_params(G):,}  D params {sum(p.numel() for p in D.parameters()):,}")

    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    l1 = nn.L1Loss()

    # referencia FID (features CISO de val/complete)
    ref_sl = []
    for cdir in sorted(VAL_COMPLETE_DIR.iterdir()):
        if not cdir.is_dir(): continue
        ci = list(cdir.glob('*_ciso.nii.gz'))
        if ci: ref_sl += _slices(nib.load(str(ci[0])).get_fdata(dtype=np.float32))
    ref_feats = _feats(ref_sl)
    print(f"  ref CISO slices (val/complete): {len(ref_sl)}")

    best_fid, hist = float('inf'), []
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        ci_it = iter(ci_loader)
        gtot = dtot = 0.0; n = 0
        for lf, mask in lf_loader:
            try: real = next(ci_it)
            except StopIteration:
                ci_it = iter(ci_loader); real = next(ci_it)
            lf, mask, real = lf.to(DEV), mask.to(DEV), real.to(DEV)
            fake = G(lf)
            # D step
            optD.zero_grad()
            d_real = D(real); d_fake = D(fake.detach())
            lossD = 0.5 * (((d_real - 1) ** 2).mean() + (d_fake ** 2).mean())
            lossD.backward(); optD.step()
            # G step
            optG.zero_grad()
            d_fake2 = D(fake)
            adv = ((d_fake2 - 1) ** 2).mean()
            anch_l1 = l1(fake * mask, lf * mask)
            anch_ssim = 1 - ssim_torch(fake, lf)
            lossG = adv + args.anchor_l1 * anch_l1 + args.anchor_ssim * anch_ssim
            lossG.backward(); optG.step()
            gtot += float(lossG.item()); dtot += float(lossD.item()); n += 1
        msg = f"Epoch {epoch:3d}/{args.epochs} | G {gtot/n:.4f} D {dtot/n:.4f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            fid = local_fid(G, ref_feats)
            flag = ''
            if fid < best_fid:
                best_fid = fid
                torch.save({'epoch': epoch, 'model_state': G.state_dict(),
                            'local_fid': fid, 'canvas': CANVAS},
                           CHECKPOINTS_DIR / 'best_1b_gan.pth')
                flag = ' [saved]'
            msg += f" | FID_local {fid:.2f}{flag}"
            hist.append({'epoch': epoch, 'fid': fid})
        print(msg, flush=True)

    print(f"\nBest FID_local: {best_fid:.2f}  (baseline ref ~81.4)")
    (RESULTS_DIR / '02_training_gan.json').write_text(json.dumps(
        {'best_local_fid': best_fid, 'history': hist,
         'anchor_l1': args.anchor_l1, 'anchor_ssim': args.anchor_ssim}, indent=2))


if __name__ == '__main__':
    main()
