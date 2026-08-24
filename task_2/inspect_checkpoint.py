"""
inspect_checkpoint.py — Inspecciona un checkpoint de nnU-Net v2

Muestra:
  - Época en que fue guardado
  - Mejor EMA dice registrado
  - Arquitectura del modelo (capas, parámetros)
  - Estado del optimizador (lr, pasos)
  - Hash MD5 del archivo (para verificar integridad futura)

Uso:
    python inspect_checkpoint.py
    python inspect_checkpoint.py --ckpt path/to/checkpoint.pth
"""

import argparse
import hashlib
import sys
from pathlib import Path

import torch

CKPT_DEFAULT = (
    r"C:\Users\rmarcar\Desktop\lisa-challenge2026\nnunet_workspace\nnUNet_results"
    r"\Dataset001_LISA_Task2\nnUNetTrainerResEncL_BoundaryLoss__nnUNetPlans__3d_fullres"
    r"\fold_0\checkpoint_final.pth"
)


def md5(path: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def count_params(state_dict: dict) -> int:
    return sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))


def print_section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def inspect(ckpt_path: Path) -> None:
    print(f"\nCargando: {ckpt_path}")
    print(f"Tamaño:   {ckpt_path.stat().st_size / 1e6:.1f} MB")
    print(f"MD5:      {md5(ckpt_path)}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # ── Claves disponibles ──────────────────────────────────────────────────
    print_section("Claves en el checkpoint")
    for k in ckpt.keys():
        val = ckpt[k]
        if isinstance(val, dict):
            print(f"  {k:30s}  dict ({len(val)} claves)")
        elif isinstance(val, (int, float, str, bool)):
            print(f"  {k:30s}  {val}")
        else:
            print(f"  {k:30s}  {type(val).__name__}")

    # ── Época ──────────────────────────────────────────────────────────────
    print_section("Progreso de entrenamiento")
    epoch = ckpt.get("current_epoch", ckpt.get("epoch", "N/A"))
    print(f"  Época guardada:          {epoch}")

    best_dice = ckpt.get("best_MA_dice_here_is_less_SE_than_the_MA_dice", None)
    if best_dice is None:
        best_dice = ckpt.get("best_MA_pseudo_dice", None)
    if best_dice is None:
        # Buscar cualquier clave que contenga 'dice' o 'best'
        for k, v in ckpt.items():
            if isinstance(v, (float, int)) and ("dice" in k.lower() or "best" in k.lower()):
                print(f"  {k}: {v}")
    else:
        print(f"  Mejor EMA pseudo-dice:   {best_dice:.6f}")

    # ── Optimizador ────────────────────────────────────────────────────────
    print_section("Optimizador")
    opt = ckpt.get("optimizer_state", ckpt.get("optimizer", None))
    if opt is not None and "param_groups" in opt:
        for i, pg in enumerate(opt["param_groups"]):
            lr   = pg.get("lr", "?")
            wd   = pg.get("weight_decay", "?")
            npar = len(pg.get("params", []))
            print(f"  Grupo {i}: lr={lr:.2e}  wd={wd}  params={npar}")
    else:
        print("  No se encontró estado del optimizador.")

    # ── Modelo ─────────────────────────────────────────────────────────────
    print_section("Modelo")
    net_sd = ckpt.get("network_weights", ckpt.get("state_dict", ckpt.get("model_state_dict", None)))
    if net_sd is not None:
        n_params = count_params(net_sd)
        n_layers = len(net_sd)
        print(f"  Parámetros totales:      {n_params:,}  (~{n_params/1e6:.1f} M)")
        print(f"  Tensores en state_dict:  {n_layers}")

        # Primeras y últimas capas
        keys = list(net_sd.keys())
        print(f"\n  Primeras 5 capas:")
        for k in keys[:5]:
            print(f"    {k:55s}  {tuple(net_sd[k].shape)}")
        print(f"\n  Últimas 5 capas:")
        for k in keys[-5:]:
            print(f"    {k:55s}  {tuple(net_sd[k].shape)}")
    else:
        print("  No se encontró state_dict del modelo.")

    # ── Conclusión ─────────────────────────────────────────────────────────
    print_section("VEREDICTO")
    epoch_ok   = isinstance(epoch, int) and epoch >= 999
    net_ok     = net_sd is not None
    params_ok  = n_params > 1e6 if net_sd is not None else False

    status = []
    status.append(f"  {'[OK]' if epoch_ok  else '[!!]'}  Epoca = {epoch}  {'(1000 epochs completados)' if epoch_ok else '(< 1000 epochs!)'}")
    status.append(f"  {'[OK]' if net_ok   else '[!!]'}  State dict del modelo presente")
    status.append(f"  {'[OK]' if params_ok else '[!!]'}  Parametros: {n_params/1e6:.1f} M  {'(modelo completo)' if params_ok else '(sospechosamente pequeno)'}")

    for s in status:
        print(s)

    if epoch_ok and net_ok and params_ok:
        print("\n  [VALID]  CHECKPOINT VALIDO -- modelo ResEncL_BoundaryLoss, Run 1, 1000 epochs")
    else:
        print("\n  [WARN]   CHECKPOINT INCOMPLETO o de RUN 2 -- revisar manualmente")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CKPT_DEFAULT, help="Ruta al archivo .pth")
    args = parser.parse_args()

    path = Path(args.ckpt)
    if not path.exists():
        print(f"[ERROR] No se encuentra: {path}")
        sys.exit(1)

    inspect(path)

    # También inspeccionar checkpoint_best si existe en el mismo directorio
    best_path = path.parent / "checkpoint_best.pth"
    if best_path.exists() and best_path != path:
        print("\n\n" + "=" * 60)
        print("  checkpoint_best.pth (para comparar)")
        print("=" * 60)
        inspect(best_path)


if __name__ == "__main__":
    main()
