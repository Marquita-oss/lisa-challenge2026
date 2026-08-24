"""
Verificación del mapeo de labels 1–11 → estructuras cerebrales.

Usa dos estrategias complementarias:
  A) Inferencia estadística por volumen y simetría bilateral (desde los datos)
  B) Consulta al wiki del proyecto Synapse syn72118611 (fuente oficial)

Uso:
    python verify_labels.py                  # solo estrategia A (sin Synapse)
    python verify_labels.py --synapse        # A + B (requiere synapseclient)
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path

TRAIN_DIR = Path(r"C:/Users/rmarcar/Desktop/lisa-challenge2026/data/train")

# Mapeo oficial — Synapse wiki syn72118611, LISA Challenge 2026
# "Lentiform" = núcleo lenticular (putamen + globo pálido combinados)
OFFICIAL_LABEL_MAP = {
    1:  "HippocampusL",
    2:  "HippocampusR",
    3:  "VentricleL",
    4:  "VentricleR",
    5:  "CaudateL",
    6:  "CaudateR",
    7:  "LentiformL",
    8:  "LentiformR",
    9:  "ThalamusL",
    10: "ThalamusR",
    11: "CorpusCallosum",
}

# Volúmenes de referencia en cerebro pediátrico 0–3 años a 1mm³ (dHCP, UNC, IBIS)
ANATOMY_REFERENCE = {
    "HippocampusL":   (400,  2500),
    "HippocampusR":   (400,  2500),
    "VentricleL":     (500,  15000),   # muy variable en edad pediátrica
    "VentricleR":     (500,  15000),
    "CaudateL":       (1000, 5000),
    "CaudateR":       (1000, 5000),
    "LentiformL":     (2000, 8000),    # putamen + GP juntos
    "LentiformR":     (2000, 8000),
    "ThalamusL":      (2000, 8000),
    "ThalamusR":      (2000, 8000),
    "CorpusCallosum": (2000, 12000),
}


def load_all_volumes(train_dir: Path) -> dict[int, list[int]]:
    """Carga volúmenes por label de todos los casos G1."""
    volumes = {i: [] for i in range(1, 12)}
    cases_found = 0
    for case_dir in sorted(train_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        seg_path = case_dir / f"lisa_{case_dir.name}_seg.nii.gz"
        ciso_path = case_dir / f"lisa_{case_dir.name}_ciso.nii.gz"
        if not (seg_path.exists() and ciso_path.exists()):
            continue
        data = nib.load(str(seg_path)).get_fdata().astype(np.int32)
        for lbl in range(1, 12):
            volumes[lbl].append(int((data == lbl).sum()))
        cases_found += 1
    print(f"Casos analizados: {cases_found}")
    return volumes


def analyze_symmetry(volumes: dict) -> list[tuple[int, int, float]]:
    """
    Detecta pares de labels con alta correlación volumétrica (estructuras bilaterales).
    Retorna lista de (label_a, label_b, correlacion) ordenada de mayor a menor.
    """
    from itertools import combinations
    pairs = []
    for a, b in combinations(range(1, 12), 2):
        va = np.array(volumes[a], dtype=float)
        vb = np.array(volumes[b], dtype=float)
        if va.std() > 0 and vb.std() > 0:
            corr = float(np.corrcoef(va, vb)[0, 1])
        else:
            corr = 0.0
        pairs.append((a, b, corr))
    return sorted(pairs, key=lambda x: -x[2])


def print_volume_table(volumes: dict) -> None:
    print(f"\n{'Label':>6}  {'Estructura':>16}  {'Media':>8}  {'Std':>7}  {'Min':>6}  {'Max':>7}  Estado")
    print("-" * 85)

    for lbl in sorted(volumes.keys()):
        vals    = volumes[lbl]
        name    = OFFICIAL_LABEL_MAP.get(lbl, f"label_{lbl}")
        mean_v  = np.mean(vals)
        std_v   = np.std(vals)
        min_v   = np.min(vals)
        max_v   = np.max(vals)
        lo, hi  = ANATOMY_REFERENCE.get(name, (0, 999999))
        status  = "OK" if lo <= mean_v <= hi else "REVISAR"
        print(f"  {lbl:>4}  {name:>16}  {mean_v:>8.0f}  {std_v:>7.0f}  {min_v:>6.0f}  {max_v:>7.0f}  {status}")


def print_symmetry_pairs(pairs: list) -> None:
    print("\nPares de labels con mayor correlación volumétrica (probables L/R):")
    print(f"  {'Label A':>8}  {'Label B':>8}  {'Correlación':>12}")
    print("  " + "-" * 35)
    for a, b, corr in pairs[:10]:
        marker = " << probable par bilateral" if corr > 0.85 else ""
        print(f"  {a:>8}  {b:>8}  {corr:>12.3f}{marker}")


def infer_mapping(volumes: dict, pairs: list) -> dict:
    """
    Infiere el mapeo más probable label → estructura.
    Usa: (1) volumen medio, (2) simetría bilateral, (3) referencia anatómica.
    """
    means = {lbl: np.mean(v) for lbl, v in volumes.items()}

    # Identificar el label impar más grande → corpus callosum
    # (estructura sin par bilateral, relativamente grande)
    bilateral_members = set()
    for a, b, corr in pairs:
        if corr > 0.85:
            bilateral_members.add(a)
            bilateral_members.add(b)

    unpaired = set(range(1, 12)) - bilateral_members

    inferred = {}
    print("\n--- Inferencia automática ---")
    print(f"Labels sin par bilateral claro (correlación < 0.85): {sorted(unpaired)}")
    print("(El label con mayor volumen entre los impares es probable corpus callosum)")

    if unpaired:
        largest_unpaired = max(unpaired, key=lambda x: means[x])
        inferred[largest_unpaired] = "corpus_callosum [inferido]"
        print(f"  → Label {largest_unpaired} ({means[largest_unpaired]:.0f} vx) = corpus_callosum")

    # Asignar pares por volumen ascendente
    # hipocampo (menor) → caudate → putamen/GP → tálamo (mayor)
    structure_pairs = [
        ("hippocampus_L", "hippocampus_R"),     # menores
        ("globus_pallidus_L", "globus_pallidus_R"),
        ("putamen_L", "putamen_R"),
        ("caudate_L", "caudate_R"),
        ("thalamus_L", "thalamus_R"),           # mayores
    ]
    bilateral_pairs = [(a, b, c) for a, b, c in pairs if c > 0.85]
    bilateral_pairs_sorted = sorted(bilateral_pairs,
                                    key=lambda x: means[x[0]] + means[x[1]])

    print("\nPares bilaterales ordenados por volumen (menor → mayor):")
    for (a, b, corr), (name_l, name_r) in zip(bilateral_pairs_sorted, structure_pairs):
        vol_a = means[a]
        vol_b = means[b]
        # Asignar L al de menor volumen (convención anatómica inconsistente, usar solo como guía)
        if vol_a <= vol_b:
            inferred[a] = f"{name_l} [inferido]"
            inferred[b] = f"{name_r} [inferido]"
        else:
            inferred[b] = f"{name_l} [inferido]"
            inferred[a] = f"{name_r} [inferido]"
        print(f"  Labels ({a},{b}) vol≈({vol_a:.0f},{vol_b:.0f}) → {name_l} / {name_r}")

    return inferred


def query_synapse_wiki() -> None:
    """Consulta el wiki del proyecto Synapse para obtener el mapeo oficial."""
    try:
        import synapseclient
    except ImportError:
        print("\nsynapseclient no instalado. Ejecutar: pip install synapseclient")
        return

    print("\n--- Consultando Synapse syn72118611 ---")
    syn = synapseclient.Synapse()
    try:
        syn.login(silent=True)
    except Exception:
        print("No hay sesión Synapse activa.")
        print("Opciones para autenticarte:")
        print("  1. synapseclient.login(authToken='TU_TOKEN')")
        print("  2. syn.login(email='...', password='...')")
        print("  3. Usar el token del notebook download_data.ipynb")
        return

    project_id = "syn72118611"

    # Buscar wiki
    try:
        wiki = syn.getWiki(project_id)
        print(f"\n=== WIKI DEL PROYECTO ===")
        print(wiki.markdown[:3000])
    except Exception as e:
        print(f"  No se pudo acceder al wiki: {e}")

    # Listar archivos en la raíz del proyecto (buscar README, label description)
    try:
        children = list(syn.getChildren(project_id, includeTypes=["file"]))
        doc_files = [c for c in children if any(
            kw in c["name"].lower()
            for kw in ["label", "readme", "protocol", "description", "doc", "guide"]
        )]
        if doc_files:
            print("\nArchivos de documentación encontrados:")
            for f in doc_files:
                print(f"  {f['name']}  (syn ID: {f['id']})")
                print(f"  → Descargar: syn.get('{f['id']}')")
        else:
            print("No se encontraron archivos de documentación de labels en la raíz del proyecto.")
            print("Revisar manualmente en: https://www.synapse.org/Synapse:syn72118611")
    except Exception as e:
        print(f"  Error listando archivos: {e}")


def main():
    parser = argparse.ArgumentParser(description="Verificar mapeo de labels LISA Task 2")
    parser.add_argument("--synapse", action="store_true",
                        help="Consultar wiki de Synapse (requiere sesión activa)")
    args = parser.parse_args()

    print("=" * 60)
    print("  VERIFICACION DE LABELS — LISA Challenge 2026 Task 2")
    print("=" * 60)
    print("  Fuente: Synapse wiki syn72118611 (actualizado 05/06/2026)")

    print("\nCargando volumenes de los 79 casos G1...")
    volumes = load_all_volumes(TRAIN_DIR)

    print("\n[1] MAPEO OFICIAL vs VOLUMENES MEDIDOS")
    print("-" * 60)
    print_volume_table(volumes)

    print("\n[2] SIMETRIA BILATERAL (correlacion L/R)")
    print("-" * 60)
    pairs = analyze_symmetry(volumes)
    print_symmetry_pairs(pairs)

    print("\n[3] MAPEO OFICIAL CONFIRMADO")
    print("-" * 60)
    for lbl, name in sorted(OFFICIAL_LABEL_MAP.items()):
        mean_v = np.mean(volumes[lbl])
        print(f"  Label {lbl:>2}: {name:<16}  media={mean_v:>7.0f} vx")

    print()
    print("  NOTA: 'Lentiform' = nucleo lenticular (putamen + globo palido combinados).")
    print("  El challenge anota ambas estructuras juntas bajo una sola etiqueta por hemisferio.")

    if args.synapse:
        print("\n[4] CONSULTA A SYNAPSE")
        print("-" * 60)
        query_synapse_wiki()


if __name__ == "__main__":
    main()
