r"""
edits.py — Lista de cambios del paper en respuesta a las revisiones LISA 2026.

Cada entrada declara a qué revisor responde:
  '1' Reviewer Atw8 · '2' Reviewer d7ui · 'b' ambos · '0' corrección propia.
y con qué comentario concreto se corresponde:
  code='R1-C1|R2-C1'  ids de review_map.py, separados por '|' si son varios.
  at=   dónde va la etiqueta clicable en paper_interactive.tex:
        'start' (por defecto), 'end', 'cap' (tras el \caption{ del bloque),
        'block' (línea propia antes de material nuevo), 'none' (sin etiqueta,
        p. ej. filas sueltas de una tabla que ya etiqueta su caption).
  what= descripción en inglés de la edición, para el índice de cambios.

make_versions.py aplica esta misma lista tres veces: sin marcar (paper_revised.tex),
marcada (paper_tracked.tex) y con navegación (paper_interactive.tex). Los números
provienen de analysis/*.json y de task_1a/results/slice_ablation*.json — ninguno se
escribió a mano.
"""


def register(edit):
    # =======================================================================
    # ABSTRACT
    # =======================================================================
    edit('b', what='abstract now previews the per-plane analysis and the slice-sampling ablation', code='R1-C1|R2-C1', note='abstract: adelanta el análisis por plano y la ablación de cortes',
         old=r"""\textbf{0.839}. For \textbf{Task~1B}""",
         new=r"""\textbf{0.839}; a per-plane analysis and a five-way slice-sampling
ablation quantify how unevenly a fixed three-slice input represents spatially
localised artifacts across acquisition orientations. For \textbf{Task~1B}""",
         tracked=r"""\textbf{0.839}\delboth{.}\addboth{; a per-plane analysis and a five-way
slice-sampling ablation quantify how unevenly a fixed three-slice input represents
spatially localised artifacts across acquisition orientations.} For \textbf{Task~1B}""")

    edit('2', what="abstract: ``makes supervision hallucinate'' softened", code='R2-C3', note='abstract: R2-C3, lenguaje',
         old=r"""correlation that makes supervision hallucinate, and a""",
         new=r"""correlation that leaves the supervisory target only partially
aligned, and a""",
         tracked=r"""correlation that \deltwo{makes supervision hallucinate}\addtwo{leaves
the supervisory target only partially aligned}, and a""")

    edit('1', what='abstract: Dice reported over the eleven annotated labels', code='R1-C2', note='abstract: R1-C2, Dice sobre 11 etiquetas',
         old=r"""a mean Dice of \textbf{0.785}, and an inference study shows that""",
         new=r"""a mean Dice of \textbf{0.784} over the eleven annotated labels
(\textbf{0.785} over the nine scored under our submission-time reading of the
criteria), and an inference study shows that""",
         tracked=r"""a mean Dice of \delone{\textbf{0.785}}\addone{\textbf{0.784} over the
eleven annotated labels (\textbf{0.785} over the nine scored under our
submission-time reading of the criteria)}, and an inference study shows that""")

    # =======================================================================
    # INTRODUCTION
    # =======================================================================
    edit('1', what='the novelty trade-off is stated up front, without defence', code='R1-W2|R1-W3', note='intro: R1-W2, declarar el trade-off de novedad de frente',
         old=r"""evaluation, validation and curation right. Concretely, our contributions are:""",
         new=r"""evaluation, validation and curation right. We state the trade-off
plainly: this paper contributes no new architecture and no new learning rule. What it
offers instead is a rigorously validated baseline together with the negative results
that delimit it --- including, as Section~\ref{sec:r1a} shows, evidence against one of
our own design choices. Concretely, our contributions are:""",
         tracked=r"""evaluation, validation and curation right. \addone{We state the
trade-off plainly: this paper contributes no new architecture and no new learning
rule. What it offers instead is a rigorously validated baseline together with the
negative results that delimit it --- including, as Section~\ref{sec:r1a} shows,
evidence against one of our own design choices.} Concretely, our contributions are:""")

    edit('b', what='contribution~1 now names the per-plane analysis and the ablation', code='R1-C1|R2-C1', note='intro: contribución 1, añade análisis por plano y ablación',
         old=r"""        out-of-fold (OOF) protocol, reaching micro-accuracy 0.839.""",
         new=r"""        out-of-fold (OOF) protocol, reaching micro-accuracy 0.839,
        together with a per-plane detection analysis and a slice-sampling ablation
        that make explicit what a fixed three-slice input does and does not capture.""",
         tracked=r"""        out-of-fold (OOF) protocol, reaching micro-accuracy 0.839\addboth{,
        together with a per-plane detection analysis and a slice-sampling ablation
        that make explicit what a fixed three-slice input does and does not capture}.""")

    edit('1', what='contribution~3: Dice over the eleven annotated labels', code='R1-C2', note='intro: contribución 3, Dice sobre 11 etiquetas',
         old=r"""        Dice 0.785 over the nine scored labels, with an inference-tuning study""",
         new=r"""        Dice 0.784 over the eleven annotated labels (0.785 over the nine
        scored under our submission-time reading), with an inference-tuning study""",
         tracked=r"""        \delone{Dice 0.785 over the nine scored labels}\addone{Dice 0.784
        over the eleven annotated labels (0.785 over the nine scored under our
        submission-time reading)}, with an inference-tuning study""")

    # =======================================================================
    # METHODS — Task 1A
    # =======================================================================
    edit('2', what='the three-slice input is justified and its assumption made explicit', code='R2-C1|R1-W1', note='métodos 1A: R2-C1, justificar los tres cortes y declarar su supuesto',
         old=r"""volumetric evidence. Each channel is windowed""",
         new=r"""volumetric evidence. This buys volumetric coverage at the fixed cost of the
three channels an ImageNet-pretrained backbone already expects, and avoids the
degenerate alternative of replicating one central slice. Its assumption is equally
explicit --- that artifact evidence is distributed along the through-plane axis enough
to be visible at three fixed positions --- and is stronger for spatially localised
artifacts (zipper, positioning, distortion) than for global ones (noise, contrast);
Section~\ref{sec:r1a} tests both. Each channel is windowed""",
         tracked=r"""volumetric evidence. \addtwo{This buys volumetric coverage at the fixed cost
of the three channels an ImageNet-pretrained backbone already expects, and avoids the
degenerate alternative of replicating one central slice. Its assumption is equally
explicit --- that artifact evidence is distributed along the through-plane axis enough
to be visible at three fixed positions --- and is stronger for spatially localised
artifacts (zipper, positioning, distortion) than for global ones (noise, contrast);
Section~\ref{sec:r1a} tests both.} Each channel is windowed""")

    edit('0', what='the published OOF estimate is declared TTA-free', code='SELF', note='métodos 1A: corrección propia, el OOF publicado no lleva TTA',
         old=r"""The final predictor averages the sigmoid outputs of all ten models
(2 backbones $\times$ 5 folds), each with eight-way dihedral test-time
augmentation~\cite{lakshminarayanan2017simple}.""",
         new=r"""The submitted predictor averages the sigmoid outputs of all ten models
(2 backbones $\times$ 5 folds), each with eight-way dihedral test-time augmentation
(TTA)~\cite{lakshminarayanan2017simple}. Every OOF estimate reported below is computed
\emph{without} TTA, matching how the stored out-of-fold predictions were generated;
Table~\ref{tab:ensemble} reports the effect of adding TTA as its own row rather than
folding it into the headline number.""",
         tracked=r"""\delself{The final predictor averages}\addself{The submitted predictor
averages} the sigmoid outputs of all ten models
(2 backbones $\times$ 5 folds), each with eight-way dihedral test-time augmentation
\addself{(TTA)}~\cite{lakshminarayanan2017simple}. \addself{Every OOF estimate reported
below is computed \emph{without} TTA, matching how the stored out-of-fold predictions
were generated; Table~\ref{tab:ensemble} reports the effect of adding TTA as its own
row rather than folding it into the headline number.}""")

    # =======================================================================
    # RESULTS — Task 1A
    # =======================================================================
    edit('2', what='per-fold spread added; the TTA claim corrected', code='R2-C2b|SELF', note='resultados 1A: R2-C2b variación entre folds + corrección del TTA',
         old=r"""Table~\ref{tab:ensemble} shows the two backbones are individually comparable
($\approx0.827$ calibrated) and that ensembling with TTA lifts OOF micro-accuracy
to 0.839, with anti-overfit calibration adding $+0.011$ over the naive $0.5$
operating point.""",
         new=r"""Table~\ref{tab:ensemble} shows the two backbones are individually comparable
($\approx0.827$ calibrated) and that ensembling lifts OOF micro-accuracy to 0.839,
with anti-overfit calibration adding $+0.012$ over the naive $0.5$ operating point.
Adding eight-way TTA on top of the ensemble does not improve the OOF estimate
($0.837$), so the gain we report comes from the ensemble and the calibration and not
from TTA. Per-fold micro-accuracy is $0.821$, $0.838$, $0.838$, $0.843$ and $0.859$
(mean $0.840$, sd $0.013$). That spread is the honest context for the rest of this
section: differences between configurations of about a point are within the
fold-to-fold variation of a 532-image cohort, and we read them as such.""",
         tracked=r"""Table~\ref{tab:ensemble} shows the two backbones are individually comparable
($\approx0.827$ calibrated) and that ensembling \delself{with TTA} lifts OOF
micro-accuracy to 0.839, with anti-overfit calibration adding
\delself{$+0.011$}\addself{$+0.012$} over the naive $0.5$ operating point.
\addself{Adding eight-way TTA on top of the ensemble does not improve the OOF estimate
($0.837$), so the gain we report comes from the ensemble and the calibration and not
from TTA.} \addtwo{Per-fold micro-accuracy is $0.821$, $0.838$, $0.838$, $0.843$ and
$0.859$ (mean $0.840$, sd $0.013$). That spread is the honest context for the rest of
this section: differences between configurations of about a point are within the
fold-to-fold variation of a 532-image cohort, and we read them as such.}""")


    # --- Tabla 1: ablación de configuración ---------------------------------
    edit('0', what='Table~1 caption: row relabelled, TTA separated, provenance stated', code='R2-C2c|SELF', at='cap', note='tab:ensemble: encabezado, color de tabla y nota de lo retirado',
         old=r"""\caption{Task~1A ensemble ablation on OOF predictions (micro-accuracy over the
flattened $\{0,1,2\}$ grid). Calibration is the anti-overfit per-class threshold
search of Section~\ref{sec:m1a}.}
\label{tab:ensemble}
\centering
\setlength{\tabcolsep}{8pt}
\begin{tabular}{lcc}""",
         new=r"""\caption{Task~1A configuration ablation on OOF predictions (micro-accuracy over
the flattened $\{0,1,2\}$ grid). Calibration is the anti-overfit per-class threshold
search of Section~\ref{sec:m1a}. TTA is listed as its own row because it belongs to
the submitted predictor but does not improve the out-of-fold estimate. Every value is
reproducible from the released OOF predictions and thresholds.}
\label{tab:ensemble}
\centering
\setlength{\tabcolsep}{8pt}
\begin{tabular}{lcc}""",
         tracked=r"""\caption{Task~1A \delself{ensemble}\addself{configuration} ablation on OOF
predictions (micro-accuracy over the flattened $\{0,1,2\}$ grid). Calibration is the
anti-overfit per-class threshold search of Section~\ref{sec:m1a}. \addself{TTA is
listed as its own row because it belongs to the submitted predictor but does not
improve the out-of-fold estimate. Every value is reproducible from the released OOF
predictions and thresholds.} \noteself{the original table labelled the ensemble row
``Ensemble (10 models) + TTA''; the stored out-of-fold predictions it reports were in
fact computed without TTA. Single-backbone calibrated scores were given as
0.827/0.827.}}
\label{tab:ensemble}
\centering
\setlength{\tabcolsep}{8pt}
\rowsself
\begin{tabular}{lcc}""")

    edit('0', what='Table~1 rows: TTA becomes its own measured row', code='SELF', at='none', note='tab:ensemble: filas',
         old=r"""EfficientNet-B4 (5 folds)      & 0.820 & 0.827 \\
ConvNeXt-Small (5 folds)       & 0.824 & 0.827 \\
Ensemble (10 models) + TTA     & \textbf{0.828} & \textbf{0.839} \\""",
         new=r"""EfficientNet-B4 (5 folds)           & 0.820 & 0.828 \\
ConvNeXt-Small (5 folds)            & 0.824 & 0.826 \\
Ensemble (10 models)                & \textbf{0.828} & \textbf{0.839} \\
Ensemble (10 models) + TTA$\times$8 & 0.827 & 0.837 \\""",
         tracked=r"""EfficientNet-B4 (5 folds)           & 0.820 & 0.828 \\
ConvNeXt-Small (5 folds)            & 0.824 & 0.826 \\
Ensemble (10 models)                & \textbf{0.828} & \textbf{0.839} \\
Ensemble (10 models) + TTA$\times$8 & 0.827 & 0.837 \\""")

    # --- Tabla 2: por clase, ahora con recall de positivos -------------------
    edit('2', what='Table~2 caption: the positive-case recall column is defined', code='R2-C2a', note='tab:perclass: R2-C2a, pie de tabla y columna nueva',
         old=r"""thresholds. $n_1,n_2$ are mild/severe plane counts; Acc.\ is three-way accuracy;
QWK is quadratic weighted $\kappa$ over $\{0,1,2\}$.}
\label{tab:perclass}
\centering
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lccccc}""",
         new=r"""thresholds. $n_1,n_2$ are mild/severe plane counts; Acc.\ is three-way accuracy;
QWK is quadratic weighted $\kappa$ over $\{0,1,2\}$; Rec.$^{+}$ is the fraction of
truly affected planes (severity $\ge1$) flagged as affected.}
\label{tab:perclass}
\centering
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccccc}""",
         tracked=r"""thresholds. $n_1,n_2$ are mild/severe plane counts; Acc.\ is three-way accuracy;
QWK is quadratic weighted $\kappa$ over $\{0,1,2\}$\addtwo{; Rec.$^{+}$ is the fraction
of truly affected planes (severity $\ge1$) flagged as affected}.
\noteself{three accuracy cells and three QWK cells differed by up to $0.02$ from what
the released out-of-fold predictions reproduce and have been recomputed; the mean is
unchanged.}}
\label{tab:perclass}
\centering
\setlength{\tabcolsep}{5pt}
\rowstwo
\begin{tabular}{lcccccc}""")

    edit('2', what='Table~2 rows: recall column added, three cells recomputed', code='R2-C2a', at='none', note='tab:perclass: filas con recall',
         old=r"""Class & $n_1$ & $n_2$ & Acc. & QWK & $(t^{(1)}, t^{(2)})$ \\
\midrule
Banding     &   9 & 12 & 0.970 & 0.671 & (0.55, 0.25) \\
Noise       &  42 & 51 & 0.908 & 0.820 & (0.65, 0.35) \\
Positioning &  45 & 25 & 0.870 & 0.455 & (0.65, 0.35) \\
Zipper      & 142 & 37 & 0.812 & 0.644 & (0.50, 0.50) \\
Motion      &  93 & 76 & 0.771 & 0.705 & (0.60, 0.25) \\
Contrast    & 166 & 31 & 0.771 & 0.608 & (0.50, 0.40) \\
Distortion  &  95 & 49 & 0.771 & 0.458 & (0.65, 0.55) \\
\midrule
\textbf{Mean} & & & \textbf{0.839} & \textbf{0.623} & --- \\""",
         new=r"""Class & $n_1$ & $n_2$ & Acc. & QWK & Rec.$^{+}$ & $(t^{(1)}, t^{(2)})$ \\
\midrule
Banding     &   9 & 12 & 0.970 & 0.690 & 0.476 & (0.55, 0.25) \\
Noise       &  42 & 51 & 0.908 & 0.813 & 0.645 & (0.65, 0.35) \\
Positioning &  45 & 25 & 0.874 & 0.477 & 0.357 & (0.65, 0.35) \\
Zipper      & 142 & 37 & 0.812 & 0.644 & 0.726 & (0.50, 0.50) \\
Motion      &  93 & 76 & 0.774 & 0.705 & 0.527 & (0.60, 0.25) \\
Contrast    & 166 & 31 & 0.771 & 0.608 & 0.716 & (0.50, 0.40) \\
Distortion  &  95 & 49 & 0.769 & 0.459 & 0.424 & (0.65, 0.55) \\
\midrule
\textbf{Mean} & & & \textbf{0.839} & \textbf{0.628} & \textbf{0.591} & --- \\""",
         tracked=r"""Class & $n_1$ & $n_2$ & Acc. & QWK & Rec.$^{+}$ & $(t^{(1)}, t^{(2)})$ \\
\midrule
Banding     &   9 & 12 & 0.970 & 0.690 & 0.476 & (0.55, 0.25) \\
Noise       &  42 & 51 & 0.908 & 0.813 & 0.645 & (0.65, 0.35) \\
Positioning &  45 & 25 & 0.874 & 0.477 & 0.357 & (0.65, 0.35) \\
Zipper      & 142 & 37 & 0.812 & 0.644 & 0.726 & (0.50, 0.50) \\
Motion      &  93 & 76 & 0.774 & 0.705 & 0.527 & (0.60, 0.25) \\
Contrast    & 166 & 31 & 0.771 & 0.608 & 0.716 & (0.50, 0.40) \\
Distortion  &  95 & 49 & 0.769 & 0.459 & 0.424 & (0.65, 0.55) \\
\midrule
\textbf{Mean} & & & \textbf{0.839} & \textbf{0.628} & \textbf{0.591} & --- \\""")

    # --- Bloque nuevo: por plano + ablación de muestreo ----------------------
    NEW_1A = r"""\paragraph{Detection by acquisition plane.}
Because the three input channels come from fixed percentile positions, an artifact
whose signature is spatially confined may be represented unevenly depending on the
orientation in which it was acquired. Table~\ref{tab:plane} therefore reports
positive-case recall --- the fraction of truly affected planes (severity $\ge1$)
flagged as affected --- per acquisition plane. Detection is markedly
orientation-dependent: 0.618 axial, 0.508 coronal and 0.629 sagittal, and of the
positives it does detect, the coronal plane assigns the correct severity in 0.362 of
cases against 0.565 axial. Per class the pattern follows the geometry of the artifact.
Zipper, a discrete band confined to few slices, is recovered in 0.86 of sagittal
positives but only 0.45 of coronal ones; banding ranges from 0.17 axial to 0.71
coronal. Positioning and distortion --- the classes with the weakest ordinal agreement
in Table~\ref{tab:perclass} --- stay below 0.52 in every orientation, so their
difficulty is representational rather than an orientation effect.

Two caveats keep this from being over-read: prevalence is unbalanced across planes
(110 of the 179 zipper positives are sagittal, against 38 axial), so part of each gap
reflects how many examples the training folds saw in that orientation; and
micro-accuracy runs opposite to recall (0.856 axial, 0.846 coronal, 0.817 sagittal)
because true negatives fill most of the grid, which is why recall is the quantity
reported here.

\begin{table}[!htbp]
\caption{Task~1A positive-case recall by acquisition plane (OOF ensemble, calibrated
thresholds). $n^{+}$ is the number of affected planes of that class in that
orientation; Rec.$^{+}$ is the fraction of them flagged as affected.}
\label{tab:plane}
\centering
\setlength{\tabcolsep}{5pt}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Axial} & \multicolumn{2}{c}{Coronal} & \multicolumn{2}{c}{Sagittal} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Class & $n^{+}$ & Rec.$^{+}$ & $n^{+}$ & Rec.$^{+}$ & $n^{+}$ & Rec.$^{+}$ \\
\midrule
Noise       &  32 & 0.78 &  24 & 0.54 &  37 & 0.59 \\
Zipper      &  38 & 0.55 &  31 & 0.45 & 110 & \textbf{0.86} \\
Positioning &  16 & 0.25 &  34 & 0.41 &  20 & 0.35 \\
Banding     &   6 & 0.17 &   7 & 0.71 &   8 & 0.50 \\
Motion      &  53 & 0.40 &  63 & 0.68 &  53 & 0.47 \\
Contrast    & 100 & 0.87 &  43 & 0.53 &  54 & 0.57 \\
Distortion  &  40 & 0.42 &  44 & 0.30 &  60 & 0.52 \\
\midrule
\textbf{All classes} & \textbf{285} & \textbf{0.618} & \textbf{246} & \textbf{0.508}
  & \textbf{342} & \textbf{0.629} \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{What the fixed three-slice input captures.}
Table~\ref{tab:slice} isolates the sampling itself. Holding the ten trained models,
the fold assignment and the calibrated thresholds fixed, we change only which slices
form the three input channels and average predicted probabilities over every triplet a
scheme produces; \emph{all slices} slides a three-slice window across the whole
through-plane axis, the densest coverage the trained backbones accept without
retraining. The design choice is vindicated in one direction and challenged in
another. Collapsing to a replicated central slice costs $0.024$ micro-accuracy and
drops positive recall from 0.591 to 0.450, so three independent positions earn their
place. But exhaustive coverage recovers precisely the artifacts that fixed sampling
was suspected of missing: zipper recall rises from 0.726 to 0.799 (axially, 0.55 to
0.74) and positioning from 0.357 to 0.400. The gain is neither free nor general ---
motion ($0.527\!\to\!0.314$) and distortion ($0.424\!\to\!0.215$) collapse under dense
sampling, consistent with signatures that are inter-slice inconsistencies and are
washed out by averaging over strongly overlapping triplets, while specificity falls
from 0.947 to 0.940, so net micro-accuracy decreases to 0.826.

The conclusion is deliberately limited: all ten models were \emph{trained} on the
25th/50th/75th-percentile input, so this measures robustness of the learned
representation to a change of sampling at inference, not what a densely trained or
fully volumetric model would achieve --- though the localised-artifact gains are
informative precisely because they survive that mismatch. What the experiment rules
out is a uniform fix: no single sampling density serves all seven classes, whose
evidence lives at different spatial scales.

\begin{table}[!htbp]
\caption{Task~1A slice-sampling ablation, inference only: same ten models, same folds,
same thresholds, different input slices. \emph{T}\ is the number of slice triplets
averaged for a 36-slice stack. The four rightmost columns are positive-case recall for
two spatially localised artifacts and two defined by inter-slice inconsistency.}
\label{tab:slice}
\centering
\setlength{\tabcolsep}{4pt}
\footnotesize
\begin{tabular}{lccccccc}
\toprule
Input slices & T & Micro-acc. & Rec.$^{+}$ & Zip. & Pos. & Mot. & Dist. \\
\midrule
Central slice $\times3$    &  1 & 0.816 & 0.450 & 0.682 & 0.314 & 0.444 & 0.153 \\
\textbf{P25/50/75 (adopted)} &  1 & \textbf{0.839} & \textbf{0.591} & 0.726 & 0.357
  & \textbf{0.527} & \textbf{0.424} \\
5 positions                &  3 & 0.834 & 0.544 & 0.754 & 0.300 & 0.491 & 0.181 \\
9 positions                &  7 & 0.832 & 0.543 & 0.704 & 0.343 & 0.473 & 0.208 \\
All slices (sliding)       & 34 & 0.826 & 0.550 & \textbf{0.799} & \textbf{0.400}
  & 0.314 & 0.215 \\
\bottomrule
\end{tabular}
\end{table}

"""

    edit('b', what='new: detection by acquisition plane, and the slice-sampling ablation', code='R1-C1|R1-W1|R2-C1|R2-C2c', at='block', note='resultados 1A: bloque nuevo (R1-C1 y R2-C1)',
         old=r"""\subsection{Task~1B --- Low-Field Enhancement}""",
         new=NEW_1A + r"""\subsection{Task~1B --- Low-Field Enhancement}""",
         tracked=(r"""\begin{revblock}{revboth}{R1+R2}""" + '\n' + NEW_1A +
                  '\\end{revblock}\n\n' + r"""\subsection{Task~1B --- Low-Field Enhancement}"""))

    # =======================================================================
    # RESULTS — Task 1B (R2-C3)
    # =======================================================================
    edit('2', what="``forced to hallucinate'' softened", code='R2-C3', note='resultados 1B: R2-C3, lenguaje',
         old=r"""so at $\approx0.5$ alignment the model is forced to hallucinate high-field texture
in slightly wrong locations.""",
         new=r"""so at $\approx0.5$ alignment the model synthesises high-field texture at
slightly displaced locations, which the reference metrics penalise.""",
         tracked=r"""so at $\approx0.5$ alignment the model \deltwo{is forced to hallucinate
high-field texture in slightly wrong locations}\addtwo{synthesises high-field texture
at slightly displaced locations, which the reference metrics penalise}.""")

    # =======================================================================
    # METHODS — Task 2 (R1-C2)
    # =======================================================================
    edit('1', what='scoring criteria restated; both aggregations announced', code='R1-C2', note='métodos 2: R1-C2, criterios de puntuación y doble agregación',
         old=r"""A scoring detail shapes every design decision: the two
ventricle labels are \emph{not} scored. The ranking is computed over the nine
remaining labels, aggregating five metrics --- $1-$DSC, HD, HD95, ASSD (mm) and
relative volume error (RVE, \%) --- with lower better and bilateral sides
averaged~\cite{maierhein2024metrics}. No external data were used.""",
         new=r"""The ranking aggregates five metrics --- $1-$DSC, HD, HD95, ASSD (mm)
and relative volume error (RVE, \%) --- with lower better and bilateral sides
averaged~\cite{maierhein2024metrics}. One scoring detail shaped our design decisions:
we built and submitted the pipeline under the reading that the two ventricle labels
were excluded, leaving nine scored labels, whereas the 2026 criteria fold them back
into the ranking. We therefore report every Task~2 result under both aggregations ---
all eleven annotated labels, and the nine of our submission-time reading --- and say
where they disagree, which as Section~\ref{sec:r2} shows is nowhere that changes a
decision. No external data were used.""",
         tracked=r"""\delone{A scoring detail shapes every design decision: the two
ventricle labels are not scored.} The ranking \delone{is computed over the nine
remaining labels, aggregating}\addone{aggregates} five metrics --- $1-$DSC, HD, HD95,
ASSD (mm) and relative volume error (RVE, \%) --- with lower better and bilateral
sides averaged~\cite{maierhein2024metrics}. \addone{One scoring detail shaped our
design decisions: we built and submitted the pipeline under the reading that the two
ventricle labels were excluded, leaving nine scored labels, whereas the 2026 criteria
fold them back into the ranking. We therefore report every Task~2 result under both
aggregations --- all eleven annotated labels, and the nine of our submission-time
reading --- and say where they disagree, which as Section~\ref{sec:r2} shows is
nowhere that changes a decision.} No external data were used.""")

    # =======================================================================
    # RESULTS — Task 2 (R1-C2)
    # =======================================================================
    edit('1', what='results open on all eleven labels, with the nine alongside', code='R1-C2', note='resultados 2: doble agregación en la frase de apertura',
         old=r"""ensemble over all 79 cases, reported over the nine scored labels. Reducing the""",
         new=r"""ensemble over all 79 cases, reported over all eleven annotated labels with the
nine-label aggregation of our submission-time reading alongside. Reducing the""",
         tracked=r"""ensemble over all 79 cases, reported over \delone{the nine scored
labels}\addone{all eleven annotated labels with the nine-label aggregation of our
submission-time reading alongside}. Reducing the""")

    POSTPROC = r"""A six-way post-processing ablation on the same ensemble makes the inference-tuning
point sharply, and Table~\ref{tab:postproc} shows it makes the same point under either
label set: the adopted structure-specific scheme (nnU-Net's automatic LCC on labels
$\{3,4,7\}$) gives the best DSC, HD95 and ASSD under both. Its story is clearest
against a uniform policy: extending LCC to every label barely moves DSC
($0.7842\!\to\!0.7806$) but inflates ASSD from 0.920 to 1.125\,mm --- the largest
single movement, in the wrong direction. At 0.064\,T the thin
tails of hippocampus and caudate are often rendered as a large body plus small,
barely-attached fragments indistinguishable from noise floaters; uniform LCC deletes
the real tail with the noise, and the surface metrics register the amputation
immediately. Three milder floater-removal variants ($<30\%$ of largest, $<50$ voxels,
and a combined rule) confirm that even gentle uniform cleanup does not pay: none beats
the adopted scheme on ASSD. Scoring the ventricles makes this check stricter rather
than looser, since they are two of the three labels the adopted scheme actually
filters; the conclusion survives it unchanged.

\begin{table}[!htbp]
\caption{Task~2 post-processing ablation on the five-fold OOF ensemble ($n=79$), under
both label sets. LCC is largest-connected-component filtering. The adopted scheme wins
on DSC, HD95 and ASSD under either aggregation; uniform LCC is the clear loser on ASSD
under both.}
\label{tab:postproc}
\centering
\setlength{\tabcolsep}{5pt}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{All 11 labels} & \multicolumn{3}{c}{9 scored (as submitted)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Post-processing & DSC\,$\uparrow$ & HD95\,$\downarrow$ & ASSD\,$\downarrow$
              & DSC\,$\uparrow$ & HD95\,$\downarrow$ & ASSD\,$\downarrow$ \\
\midrule
None (raw ensemble)            & 0.7836 & 2.478 & 0.938 & 0.7843 & 2.564 & 0.999 \\
\textbf{Per-label (adopted)}   & \textbf{0.7842} & \textbf{2.401} & \textbf{0.920}
  & \textbf{0.7849} & \textbf{2.483} & \textbf{0.978} \\
LCC on all labels              & 0.7806 & 2.464 & 1.125 & 0.7805 & 2.560 & 1.228 \\
Fragments $<30\%$ of largest   & 0.7835 & 2.490 & 1.001 & 0.7842 & 2.583 & 1.074 \\
Fragments $<50$ voxels         & 0.7835 & 2.491 & 0.997 & 0.7842 & 2.582 & 1.069 \\
Adopted $+<50$ voxels          & 0.7841 & 2.416 & 0.977 & 0.7848 & 2.501 & 1.047 \\
\bottomrule
\end{tabular}
\end{table}"""

    edit('1', what='post-processing ablation rewritten under both label sets, with a new table', code='R1-C2', note='resultados 2: párrafo de post-proceso + tabla nueva con ambas agregaciones',
         old=r"""A six-way post-processing ablation on the same ensemble makes the inference-tuning
point sharply. The adopted structure-specific scheme (nnU-Net's automatic LCC on
labels $\{3,4,7\}$) gives the best DSC (0.7849), HD95 (2.483\,mm) and ASSD
(0.978\,mm). Its story is clearest against a uniform policy: extending LCC to all
eleven labels barely moves DSC ($0.7849\!\to\!0.7805$) but inflates ASSD from
0.978 to 1.228\,mm --- the largest single movement, in the wrong direction. At
0.064\,T the thin tails of hippocampus and caudate are often rendered as a large
body plus small, barely-attached fragments indistinguishable from noise floaters;
uniform LCC deletes the real tail with the noise, and the surface metrics register
the amputation immediately. Three milder floater-removal variants ($<30\%$ of
largest, $<50$ voxels, and a combined rule) confirm even gentle uniform cleanup
does not pay: none beats the adopted scheme on ASSD.""",
         new=POSTPROC,
         tracked=(r"""\noteone{the original paragraph reported this ablation only over the
nine labels of our submission-time reading and carried no table.}"""
                  + '\n' + r'\begin{revblock}{revone}{R1}' + '\n' + POSTPROC + '\n\\end{revblock}'))

    edit('1', what='per-structure caption: the eleven-label mean leads', code='R1-C2', at='cap', note='tab:perstruct: pie de tabla',
         old=r"""\caption{Task~2 per-structure performance of the five-fold ensemble with the
adopted post-processing ($n=79$). The nine scored labels determine the ranking;
the two ventricle rows are shown for completeness only and do not enter the mean.}""",
         new=r"""\caption{Task~2 per-structure performance of the five-fold ensemble with the
adopted post-processing ($n=79$). All eleven annotated labels enter the primary mean;
the mean over the nine labels of our submission-time reading is given below it for
continuity. Ranking under either set puts the same structures at the extremes.}""",
         tracked=r"""\caption{Task~2 per-structure performance of the five-fold ensemble with the
adopted post-processing ($n=79$). \delone{The nine scored labels determine the ranking;
the two ventricle rows are shown for completeness only and do not enter the mean.}
\addone{All eleven annotated labels enter the primary mean; the mean over the nine
labels of our submission-time reading is given below it for continuity. Ranking under
either set puts the same structures at the extremes.}}""")

    PERSTRUCT = r"""Hippocampus L & 0.627 & 5.295 & 3.653 & 1.709 & 22.95 \\
Hippocampus R & 0.654 & 4.974 & 3.515 & 1.397 & 24.03 \\
Ventricle L   & 0.790 & 4.786 & 1.913 & 0.643 & 13.55 \\
Ventricle R   & 0.772 & 5.155 & 2.153 & 0.678 & 12.70 \\
Caudate L     & 0.812 & 4.286 & 2.319 & 0.861 & 14.27 \\
Caudate R     & 0.802 & 3.766 & 2.129 & 0.818 & 11.53 \\
Lentiform L   & 0.837 & 5.181 & 2.240 & 1.024 & 13.57 \\
Lentiform R   & 0.833 & 4.174 & 2.695 & 1.101 & 16.80 \\
Thalamus L    & 0.876 & 3.813 & 2.191 & 0.763 & 11.14 \\
Thalamus R    & 0.875 & 4.072 & 2.080 & 0.666 & \phantom{0}8.79 \\
Corpus callosum & 0.749 & 4.989 & 1.520 & 0.459 & \phantom{0}7.74 \\
\midrule
\textbf{Mean (11 labels)} & \textbf{0.7842} & \textbf{4.590} & \textbf{2.401} &
  \textbf{0.920} & \textbf{14.28} \\
\textit{Mean (9, as submitted)} & \textit{0.7849} & \textit{4.505} & \textit{2.483} &
  \textit{0.978} & \textit{14.53} \\
\bottomrule"""

    edit('1', what='per-structure rows: ventricles moved into the table, two mean rows', code='R1-C2', at='none', note='tab:perstruct: ventrículos dentro de la tabla y dos filas de media',
         old=r"""Hippocampus L & 0.627 & 5.295 & 3.653 & 1.709 & 22.95 \\
Hippocampus R & 0.654 & 4.974 & 3.515 & 1.397 & 24.03 \\
Caudate L     & 0.812 & 4.286 & 2.319 & 0.861 & 14.27 \\
Caudate R     & 0.802 & 3.766 & 2.129 & 0.818 & 11.53 \\
Lentiform L   & 0.837 & 5.181 & 2.240 & 1.024 & 13.57 \\
Lentiform R   & 0.833 & 4.174 & 2.695 & 1.101 & 16.80 \\
Thalamus L    & 0.876 & 3.813 & 2.191 & 0.763 & 11.14 \\
Thalamus R    & 0.875 & 4.072 & 2.080 & 0.666 & \phantom{0}8.79 \\
Corpus callosum & 0.749 & 4.989 & 1.520 & 0.459 & \phantom{0}7.74 \\
\midrule
\textbf{Mean (9 scored)} & \textbf{0.7849} & \textbf{4.505} & \textbf{2.483} &
  \textbf{0.978} & \textbf{14.53} \\
\midrule
\multicolumn{6}{l}{\small\itshape Non-scored (reference only)} \\
Ventricle L   & 0.790 & 4.786 & 1.913 & 0.643 & 13.55 \\
Ventricle R   & 0.772 & 5.155 & 2.153 & 0.678 & 12.70 \\
\bottomrule""",
         new=PERSTRUCT, tracked=PERSTRUCT)

    edit('1', what='per-structure table colouring (tracked copy only)', code='', at='none', note='tab:perstruct: color de tabla en la versión marcada',
         old=r"""\label{tab:perstruct}
\centering
\setlength{\tabcolsep}{5pt}""",
         new=r"""\label{tab:perstruct}
\centering
\setlength{\tabcolsep}{5pt}""",
         tracked=r"""\label{tab:perstruct}
\centering
\setlength{\tabcolsep}{5pt}
\rowsone""")

    edit('1', what='per-structure paragraph: both means quoted', code='R1-C2', note='resultados 2: párrafo por estructura, medias actualizadas',
         old=r"""Table~\ref{tab:perstruct} gives the per-structure breakdown. The hippocampus is
the bottleneck by a wide margin (both sides near Dice 0.64, worst surface
distances and volume errors), while the thalamus is the ceiling (Dice
0.876/0.875).""",
         new=r"""Table~\ref{tab:perstruct} gives the per-structure breakdown, over all eleven
labels (mean Dice 0.7842) and over the nine of our submission-time reading (0.7849);
the two differ by less than a thousandth, because the ventricles sit close to the
cohort mean. The hippocampus is the bottleneck by a wide margin (both sides near Dice
0.64, worst surface distances and volume errors), while the thalamus is the ceiling
(Dice 0.876/0.875).""",
         tracked=r"""Table~\ref{tab:perstruct} gives the per-structure breakdown\addone{, over
all eleven labels (mean Dice 0.7842) and over the nine of our submission-time reading
(0.7849); the two differ by less than a thousandth, because the ventricles sit close
to the cohort mean}. The hippocampus is
the bottleneck by a wide margin (both sides near Dice 0.64, worst surface
distances and volume errors), while the thalamus is the ceiling (Dice
0.876/0.875).""")

    edit('1', what='figure caption: ventricles are scored under the 2026 criteria', code='R1-C2', note='figura: los ventrículos ya no son "no puntuados"',
         old=r"""Columns: CISO volume, ground truth, prediction (ventricles drawn
but not scored).""",
         new=r"""Columns: CISO volume, ground truth, prediction (ventricles drawn,
and scored under the 2026 criteria).""",
         tracked=r"""Columns: CISO volume, ground truth, prediction (ventricles drawn\delone{
but not scored}\addone{, and scored under the 2026 criteria}).""")

    # =======================================================================
    # DISCUSSION
    # =======================================================================
    edit('2', what="``makes losses hallucinate'' softened", code='R2-C3', note='discusión: R2-C3, lenguaje',
         old=r"""makes pixel and feature losses hallucinate, and proxy-driven checkpoint selection""",
         new=r"""makes pixel and feature losses reward texture which is plausible but
spatially misplaced, and proxy-driven checkpoint selection""",
         tracked=r"""\deltwo{makes pixel and feature losses hallucinate}\addtwo{makes pixel
and feature losses reward texture which is plausible but spatially misplaced}, and
proxy-driven checkpoint selection""")

    edit('1', what='ASSD figures restated over the eleven labels', code='R1-C2', note='discusión: ASSD sobre las once etiquetas',
         old=r"""to small filamentous gray matter deletes real anatomy (ASSD
$0.978\!\to\!1.228$\,mm), so the right move was to trust nnU-Net's per-label
selection.""",
         new=r"""to small filamentous gray matter deletes real anatomy (ASSD
$0.920\!\to\!1.125$\,mm over all eleven labels), so the right move was to trust
nnU-Net's per-label selection.""",
         tracked=r"""to small filamentous gray matter deletes real anatomy
\delone{\mbox{(ASSD $0.978\to1.228$\,mm)}}\addone{(ASSD $0.920\!\to\!1.125$\,mm over
all eleven labels)}, so the right move was to trust nnU-Net's per-label selection.""")

    SLICE_DISC = r"""\paragraph{Where our own design choice does not hold up.}
The slice-sampling ablation of Table~\ref{tab:slice} is a negative result about this
paper rather than about the field. Three fixed percentile slices clearly beat a single
central slice, so the input is not decorative; but they under-detect precisely the
artifacts whose signature is spatially confined, and exhaustive coverage recovers much
of that loss for zipper and positioning while destroying it for motion and distortion.
The remaining headroom in Task~1A therefore sits in the input representation rather
than in the architecture --- consistent with this paper's thesis, uncomfortable for its
own Task~1A design --- and no uniform sampling rule can serve seven classes whose
evidence lives at different spatial scales. We report it because a baseline paper earns
its place only if it also marks where its own baseline is weak.

"""

    edit('b', what='new: what the sampling result says about our own design choice', code='R1-W1|R2-C1', at='block', note='discusión: párrafo nuevo sobre el hallazgo de muestreo',
         old=r"""\paragraph{Limitations and future work.}""",
         new=SLICE_DISC + r"""\paragraph{Limitations and future work.}""",
         tracked=(r'\begin{revblock}{revboth}{R1+R2}' + '\n' + SLICE_DISC +
                  '\\end{revblock}\n\n' + r"""\paragraph{Limitations and future work.}"""))

    edit('b', what='future work: class-conditioned slice selection, dense volumetric encoder', code='R1-W1|R2-C1', note='trabajo futuro: selección de cortes condicionada a la clase',
         old=r"""symmetric-label recipe for the thin-tail hippocampal failures in Task~2; and
higher-resolution features plus validation of the label-free threshold transport
for Task~1A.""",
         new=r"""symmetric-label recipe for the thin-tail hippocampal failures in Task~2;
and, for Task~1A, a slice-selection rule conditioned on the artifact class --- or a
densely sampled, end-to-end volumetric encoder, which our inference-only ablation
cannot stand in for --- together with higher-resolution features and validation of the
label-free threshold transport.""",
         tracked=r"""symmetric-label recipe for the thin-tail hippocampal failures in Task~2;
and\addboth{, for Task~1A, a slice-selection rule conditioned on the artifact class ---
or a densely sampled, end-to-end volumetric encoder, which our inference-only ablation
cannot stand in for --- together with} \delboth{higher-resolution features plus
validation of the label-free threshold transport for Task~1A}\addboth{higher-resolution
features and validation of the label-free threshold transport}.""")

    # =======================================================================
    # Recortes editoriales para absorber el material nuevo sin inflar el paper
    # =======================================================================
    edit('0', what="editorial trim: repeated ``0.064\\,T'' removed", code='SELF', note='discusión: quita la repetición de "0.064 T" y comprime el cierre',
         old=r"""and the Task~2 hippocampus --- small, thin, curved and weakly contrasted at
0.064\,T --- never closed its early gap, at Dice levels the physics of 0.064\,T
predicts, indicating signal-limited rather than method-limited pipelines.""",
         new=r"""and the Task~2 hippocampus --- small, thin, curved and weakly contrasted at
0.064\,T --- never closed its early gap, at Dice levels the physics predicts,
indicating signal-limited rather than method-limited pipelines.""",
         tracked=r"""and the Task~2 hippocampus --- small, thin, curved and weakly contrasted at
0.064\,T --- never closed its early gap, at Dice levels the physics
\delself{of 0.064\,T} predicts, indicating signal-limited rather than method-limited
pipelines.""")

    edit('0', what='editorial trim: closing sentence compressed', code='SELF', note='discusión: comprime el arranque, ya dicho en abstract e introducción',
         old=r"""Across three tasks the same lesson recurs: at 0.064\,T a well-configured, honestly
validated pipeline is competitive, and the reliable gains come from rigour rather
than novelty.""",
         new=r"""Across three tasks the same lesson recurs: at 0.064\,T a well-configured,
honestly validated pipeline is competitive.""",
         tracked=r"""Across three tasks the same lesson recurs: at 0.064\,T a well-configured,
honestly validated pipeline is competitive\delself{, and the reliable gains come from
rigour rather than novelty}.""")

    edit('0', what='editorial trim: redundant pass-through sentence compressed', code='SELF', note='resultados 1B: comprime una frase redundante del control de pass-through',
         old=r"""On 12 held-out complete subjects the identity (native pass-through) is the FID
optimum among fixed operators: unsharp masking, gamma, CLAHE and
denoise-then-sharpen all \emph{increase} per-subject FID (from $138.7$ for
identity up to $151.7$ for aggressive unsharp) and BRISQUE, so the
percentile-normalised native image already sits at a joint optimum.""",
         new=r"""On 12 held-out complete subjects the identity (native pass-through) is the FID
optimum among fixed operators: unsharp masking, gamma, CLAHE and denoise-then-sharpen
all \emph{increase} per-subject FID (from $138.7$ up to $151.7$) and BRISQUE, so the
percentile-normalised native image already sits at a joint optimum.""",
         tracked=r"""On 12 held-out complete subjects the identity (native pass-through) is the FID
optimum among fixed operators: unsharp masking, gamma, CLAHE and denoise-then-sharpen
all \emph{increase} per-subject FID (from $138.7$ \delself{for identity} up to $151.7$
\delself{for aggressive unsharp}) and BRISQUE, so the percentile-normalised native
image already sits at a joint optimum.""")

    edit('0', what='post-processing table: column spacing fixed to stop overflow', code='SELF', at='none', note='tab:postproc: evita el desbordamiento de caja',
         old=r"""\setlength{\tabcolsep}{5pt}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{All 11 labels} & \multicolumn{3}{c}{9 scored (as submitted)} \\""",
         new=r"""\setlength{\tabcolsep}{3.5pt}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{All 11 labels} & \multicolumn{3}{c}{9 scored (as submitted)} \\""",
         tracked=r"""\setlength{\tabcolsep}{3.5pt}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{All 11 labels} & \multicolumn{3}{c}{9 scored (as submitted)} \\""")


    edit('1', what='abstract: Task~2 described as eleven annotated labels', code='R1-C2', note='abstract: la descripción de Task 2 ya no puede decir "nine scored"',
         old=r"""\textbf{Task~2} (3D segmentation of nine scored subcortical structures) a""",
         new=r"""\textbf{Task~2} (3D segmentation of eleven annotated subcortical labels) a""",
         tracked=r"""\textbf{Task~2} (3D segmentation of \delone{nine scored}\addone{eleven
annotated} subcortical \delone{structures}\addone{labels}) a""")

    edit('1', what='intro: Task~2 described as eleven annotated labels', code='R1-C2', note='intro: descripción de Task 2',
         old=r"""for 3D segmentation of subcortical gray matter, scored over nine structures.""",
         new=r"""for 3D segmentation of subcortical gray matter over eleven annotated labels.""",
         tracked=r"""for 3D segmentation of subcortical gray matter\delone{, scored over nine
structures}\addone{ over eleven annotated labels}.""")

    edit('0', what='QWK values quoted in prose follow the recomputed table', code='SELF', note='resultados 1A: los QWK citados en prosa deben seguir a la tabla recomputada',
         old=r"""is highest for Noise (0.82) and Motion (0.71) and lowest for
Positioning (0.46) and Distortion (0.46) --- the artifacts with the least
stereotyped spatial signature, and the natural targets for future work.""",
         new=r"""is highest for Noise (0.81) and Motion (0.71) and lowest for
Positioning (0.48) and Distortion (0.46) --- the artifacts with the least
stereotyped spatial signature, and the natural targets for future work.""",
         tracked=r"""is highest for Noise (\delself{0.82}\addself{0.81}) and Motion (0.71) and
lowest for Positioning (\delself{0.46}\addself{0.48}) and Distortion (0.46) --- the
artifacts with the least stereotyped spatial signature, and the natural targets for
future work.""")
