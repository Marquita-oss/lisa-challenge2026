"""
task2_label_sets.py — Task 2 puntuado sobre 9 y sobre 11 etiquetas.

El revisor 1 señala que los criterios 2026 sí puntúan los ventrículos, mientras que la
submission se diseñó bajo la lectura de nueve etiquetas. Este script reagrega el barrido
de post-procesado ya calculado (métricas por etiqueta, n=79, ensemble OOF de 5 folds) bajo
ambos conjuntos, para poder afirmar si la decisión de post-procesado cambia o no.

Etiquetas: 1,2 hipocampo · 3,4 ventrículo lateral · 5,6 caudado · 7,8 lentiforme ·
           9,10 tálamo · 11 cuerpo calloso.  Las 9 "scored" excluyen 3 y 4.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SWEEP = REPO / 'task_2' / 'qc_output' / 'postproc_sweep.json'

SCORED9 = [1, 2, 5, 6, 7, 8, 9, 10, 11]
ALL11 = list(range(1, 12))
METRICS = ['dice', 'hd', 'hd95', 'assd', 'rve']
NAMES = {1: 'Hippocampus L', 2: 'Hippocampus R', 3: 'Ventricle L', 4: 'Ventricle R',
         5: 'Caudate L', 6: 'Caudate R', 7: 'Lentiform L', 8: 'Lentiform R',
         9: 'Thalamus L', 10: 'Thalamus R', 11: 'Corpus callosum'}
VARIANTS = {'raw': 'no post-processing', 'pkl': 'nnU-Net structure-specific (adopted)',
            'lcc_all': 'largest component, all labels', 'frac30': 'drop fragments <30% of largest',
            'abs50': 'drop fragments <50 voxels', 'pkl+abs50': 'adopted + <50 voxels'}


def mean_over(per_label, variant, labels, metric):
    return sum(per_label[variant][str(l)][metric] for l in labels) / len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default=str(SWEEP),
                    help='postproc_sweep.json producido por task_2/eval_postproc_sweep.py')
    args = ap.parse_args()

    sweep_path = Path(args.sweep)
    sweep = json.loads(sweep_path.read_text())
    per = sweep['per_label']

    out = {'source': sweep_path.name, 'n_cases': 79,
           'note': 'ensemble OOF de 5 folds contra ground truth manual; medias no ponderadas '
                   'sobre etiquetas, lados bilaterales incluidos por separado',
           'postproc': {}, 'per_structure_adopted': {}}

    for v in per:
        out['postproc'][v] = {
            'description': VARIANTS.get(v, v),
            'scored9': {m: round(mean_over(per, v, SCORED9, m), 4) for m in METRICS},
            'all11': {m: round(mean_over(per, v, ALL11, m), 4) for m in METRICS},
        }

    for l in ALL11:
        out['per_structure_adopted'][NAMES[l]] = {
            m: round(per['pkl'][str(l)][m], 4) for m in METRICS}

    a = out['postproc']['pkl']
    out['headline'] = {
        'adopted_dice_all11': a['all11']['dice'], 'adopted_dice_scored9': a['scored9']['dice'],
        'adopted_assd_all11': a['all11']['assd'], 'adopted_assd_scored9': a['scored9']['assd'],
        'uniform_lcc_assd_all11': out['postproc']['lcc_all']['all11']['assd'],
        'conclusion': 'el esquema adoptado es el mejor en DSC, HD95 y ASSD bajo ambos '
                      'conjuntos de etiquetas; el LCC uniforme degrada ASSD en ambos',
    }

    (HERE / 'task2_label_sets.json').write_text(json.dumps(out, indent=2))

    print(f"{'variant':34s} {'DSC9':>7s} {'DSC11':>7s} {'HD95_9':>7s} {'HD95_11':>8s} "
          f"{'ASSD9':>7s} {'ASSD11':>7s} {'RVE9':>6s} {'RVE11':>6s}")
    for v, d in out['postproc'].items():
        s, a11 = d['scored9'], d['all11']
        print(f"{d['description'][:34]:34s} {s['dice']:7.4f} {a11['dice']:7.4f} {s['hd95']:7.3f} "
              f"{a11['hd95']:8.3f} {s['assd']:7.3f} {a11['assd']:7.3f} {s['rve']:6.2f} {a11['rve']:6.2f}")
    print("\nGuardado:", HERE / 'task2_label_sets.json')


if __name__ == '__main__':
    main()
