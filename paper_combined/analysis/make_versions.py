"""
make_versions.py — Genera las tres versiones del paper desde paper.tex.

Las tres salen de la MISMA lista de ediciones (edits.py), así que no pueden divergir:
  - paper_revised.tex     : texto final limpio (camera-ready).
  - paper_tracked.tex     : lo eliminado tachado y lo añadido en color, azul para el
                            revisor 1 (Atw8) y naranja para el revisor 2 (d7ui).
  - paper_interactive.tex : texto limpio + una etiqueta clicable en cada cambio que
                            salta al comentario del revisor en el apéndice, y de vuelta.
                            Los ids de comentario viven en review_map.py.

Uso:  python make_versions.py [--check]
      --check verifica que al quitar las marcas del tracked se recupera el revised.
"""
import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'paper.tex'
OUT_CLEAN = HERE.parent / 'paper_revised.tex'
OUT_TRACK = HERE.parent / 'paper_tracked.tex'
OUT_INTER = HERE.parent / 'paper_interactive.tex'

EDITS = []


def edit(rev, old, new, tracked=None, note='', code='', at='start', what=''):
    """rev:  '1' | '2' | 'b' (ambos) | '0' (corrección propia, sin revisor).
    code: ids de review_map.py separados por '|' ('R1-C1|R2-C1').
    at:   colocación de la etiqueta en la versión interactiva —
          'start' | 'end' | 'cap' | 'block' | 'none'.
    what: descripción en inglés para el índice de cambios del apéndice.
    what: descripción en inglés para el índice de cambios del apéndice."""
    EDITS.append({'rev': rev, 'old': old, 'new': new, 'tracked': tracked, 'note': note,
                  'code': [c.strip() for c in code.split('|') if c.strip()], 'at': at,
                  'what': what or note})


CMD = {'1': ('addone', 'delone'), '2': ('addtwo', 'deltwo'),
       'b': ('addboth', 'delboth'), '0': ('addself', 'delself')}
TAG = {'1': 'R1', '2': 'R2', 'b': 'R1+R2', '0': 'self'}
COLOR = {'1': 'revone', '2': 'revtwo', 'b': 'revboth', '0': 'revself'}


def default_tracked(rev, old, new):
    add, dele = CMD[rev]
    parts = []
    if old.strip():
        parts.append('\\%s{%s}' % (dele, old.strip()))
    if new.strip():
        parts.append('\\%s{%s}' % (add, new.strip()))
    return ' '.join(parts)


PREAMBLE_TRACK = r"""
% ---------------------------------------------------------------------------
% Marcado de cambios (versión con control de cambios, no para camera-ready)
% ---------------------------------------------------------------------------
\usepackage[normalem]{ulem}
\definecolor{revone}{HTML}{1B5FBF}   % Reviewer 1 (Atw8)
\definecolor{revtwo}{HTML}{C2410C}   % Reviewer 2 (d7ui)
\definecolor{revboth}{HTML}{7C3AED}  % ambos revisores
\definecolor{revself}{HTML}{047857}  % corrección propia detectada al revisar
\newcommand{\addone}[1]{\textcolor{revone}{#1}}
\newcommand{\delone}[1]{\textcolor{revone}{\sout{#1}}}
\newcommand{\addtwo}[1]{\textcolor{revtwo}{#1}}
\newcommand{\deltwo}[1]{\textcolor{revtwo}{\sout{#1}}}
\newcommand{\addboth}[1]{\textcolor{revboth}{#1}}
\newcommand{\delboth}[1]{\textcolor{revboth}{\sout{#1}}}
\newcommand{\addself}[1]{\textcolor{revself}{#1}}
\newcommand{\delself}[1]{\textcolor{revself}{\sout{#1}}}
% Para bloques largos (tablas, párrafos enteros) el tachado de ulem es frágil:
% se anota qué se quitó, en color, en vez de tacharlo.
\newcommand{\noteone}[1]{{\color{revone}\small\textbf{[\,R1 $\cdot$ removed\,]}\ \textit{#1}\par}}
\newcommand{\notetwo}[1]{{\color{revtwo}\small\textbf{[\,R2 $\cdot$ removed\,]}\ \textit{#1}\par}}
\newcommand{\noteboth}[1]{{\color{revboth}\small\textbf{[\,R1+R2 $\cdot$ removed\,]}\ \textit{#1}\par}}
\newcommand{\noteself}[1]{{\color{revself}\small\textbf{[\,self $\cdot$ removed\,]}\ \textit{#1}\par}}
% Colorea una tabla entera: va antes de \begin{tabular}, dentro del grupo del float.
\newcommand{\rowsone}{\color{revone}}
\newcommand{\rowstwo}{\color{revtwo}}
\newcommand{\rowsboth}{\color{revboth}}
\newcommand{\rowsself}{\color{revself}}
\newenvironment{revblock}[2]{\par\color{#1}\noindent\textbf{[\,#2\,]}\ \ignorespaces}{\par}
"""

LEGEND = r"""
\begin{center}\small
\fbox{\begin{minipage}{0.93\textwidth}
\textbf{Change-tracking legend.} This copy marks every revision made in response to review.
\textcolor{revone}{Blue = Reviewer~1 (Atw8)}: per-plane recall, slice-selection evidence,
ventricles folded into the Task~2 ranking, framing of the contribution.
\textcolor{revtwo}{Orange = Reviewer~2 (d7ui)}: justification of the three-slice input,
per-fold and per-artifact detail, more cautious wording.
\textcolor{revboth}{Purple = both reviewers.}
\textcolor{revself}{Green = correction we found ourselves while preparing the revision.}
Struck-through text was removed; coloured text was added. Blocks are additionally labelled
\textbf{[\,R1\,]}, \textbf{[\,R2\,]}, \textbf{[\,R1+R2\,]} or \textbf{[\,self\,]} so the copy
reads the same in black and white. The clean version is \texttt{paper\_revised.tex}.
\end{minipage}}
\end{center}
"""


# ---------------------------------------------------------------------------
# VERSIÓN INTERACTIVA
# ---------------------------------------------------------------------------
PREAMBLE_INTER = r"""
% ---------------------------------------------------------------------------
% Navegación cambio <-> comentario del revisor (copia interactiva, no camera-ready)
% ---------------------------------------------------------------------------
\usepackage{longtable}
\definecolor{revone}{HTML}{1B5FBF}   % Reviewer 1 (Atw8)
\definecolor{revtwo}{HTML}{C2410C}   % Reviewer 2 (d7ui)
\definecolor{revboth}{HTML}{7C3AED}  % ambos revisores
\definecolor{revself}{HTML}{047857}  % corrección propia detectada al revisar
\definecolor{chgback}{HTML}{6D28D9}  % enlaces de vuelta, apéndice -> cambio
\definecolor{revdel}{HTML}{6B7280}   % texto retirado, en el apéndice B
\DeclareRobustCommand{\chgback}[1]{\protect\hyperlink{chg:#1}{\textcolor{chgback}{back to the text}}}
% \chgc: el id del comentario, clicable hacia el apéndice.
\DeclareRobustCommand{\chgc}[1]{\protect\hyperlink{rc:#1}{#1}}
% \chgtag{color}{n}{ids}: marca en superíndice al lado del texto cambiado.
\DeclareRobustCommand{\chgtag}[3]{%
  \hypertarget{chg:#2}{}\label{chgl:#2}%
  \textsuperscript{\textcolor{#1}{\sffamily\bfseries\fontsize{5.5}{6}\selectfont%
   [#2$\vert$#3]}}}
% \chgblocktag: lo mismo pero en línea propia, para material enteramente nuevo.
\DeclareRobustCommand{\chgblocktag}[3]{%
  \par\smallskip\noindent\hypertarget{chg:#2}{}\label{chgl:#2}%
  {\sffamily\bfseries\fontsize{7}{8}\selectfont\textcolor{#1}{%
   [#2$\vert$#3]}}%
  \par\nobreak\smallskip}
% \chgref: enlace de vuelta desde el apéndice hasta un cambio concreto.
\DeclareRobustCommand{\chgref}[1]{\protect\hyperlink{chg:#1}{\textcolor{chgback}{\textbf{#1}}}~(p.~\pageref{chgl:#1})}
"""

NAVBOX = r"""
\begin{center}\small
\fbox{\begin{minipage}{0.93\textwidth}
\textbf{Interactive review copy.} Every revision made in response to review is shown
\emph{in place}: \sout{struck-through} is what the submitted paper said, coloured text is what
it says now, in the colour of the reviewer who asked ---
\textcolor{revone}{blue = Reviewer~1 (Atw8)}, \textcolor{revtwo}{orange = Reviewer~2 (d7ui)},
\textcolor{revboth}{purple = both}, \textcolor{revself}{green = a correction we found
ourselves}. Nothing has to be compared against another file.
\textbf{What is clickable are the comments.} Each change carries a tag such as
\textsuperscript{\textcolor{revone}{\sffamily\bfseries\fontsize{5.5}{6}\selectfont[7$\vert$R1-C2]}};
click an id in it to read the reviewer comment it answers, quoted verbatim with our response,
in Appendix~A --- and every comment there links back to each change that answers it, with page
numbers. Blocks are also labelled \textbf{[\,R1\,]}, \textbf{[\,R2\,]}, \textbf{[\,R1+R2\,]}
or \textbf{[\,self\,]} so the copy reads the same in black and white.
The clean camera-ready text is \texttt{paper\_revised.pdf}.
\end{minipage}}
\end{center}
"""


SECPAT = re.compile(r'\\(section|subsection)\{([^}]*)\}')


def locate(src, pos):
    """Etiqueta legible de dónde cae una edición: '§3.1 Results — Task 1A'."""
    ab = src.find(r'\begin{abstract}')
    ae = src.find(r'\end{abstract}')
    if ab != -1 and ab < pos < ae:
        return 'Abstract'
    sec = sub = 0
    label = 'Front matter'
    for m in SECPAT.finditer(src, 0, pos):
        title = m.group(2).replace('~', ' ').replace('---', '---')
        if m.group(1) == 'section':
            sec, sub = sec + 1, 0
            label = '\\S%d %s' % (sec, title)
        else:
            sub += 1
            label = '\\S%d.%d %s' % (sec, sub, title)
    return label


def number_edits(src):
    """Numera los cambios en orden de aparición en el documento.

    Devuelve (nums, rows): nums[i] es el número del cambio i (o None si esa
    edición no lleva etiqueta), rows son los cambios numerados en orden de lectura.
    """
    taggable = [i for i, e in enumerate(EDITS) if e['code'] and e['at'] != 'none'
                and e['new'] != e['old']]
    taggable.sort(key=lambda i: src.index(EDITS[i]['old']))
    nums = [None] * len(EDITS)
    rows = []
    for n, i in enumerate(taggable, 1):
        nums[i] = n
        e = EDITS[i]
        rows.append({'n': n, 'rev': e['rev'], 'code': e['code'], 'what': e['what'],
                     'where': locate(src, src.index(e['old']))})
    rows.sort(key=lambda r: r['n'])
    return nums, rows


def inter_tag(e, n):
    body = ',\\,'.join('\\chgc{%s}' % c for c in e['code'])
    return '\\chgtag{%s}{%d}{%s}' % (COLOR[e['rev']], n, body)


def inter_replacement(e, n):
    """Marcado antes/después en el propio párrafo + etiqueta clicable.

    El cuerpo es el mismo de paper_tracked.tex --- lo retirado tachado y lo
    añadido en color --- para que la comparación se lea en el sitio, sin saltar
    a ningún lado. Lo único que se navega con el ratón son las etiquetas.
    """
    new = e['tracked'] if e['tracked'] is not None else \
        default_tracked(e['rev'], e['old'], e['new'])
    if n in (None, 0) or not e['code']:
        return new
    if e['at'] == 'block':
        body = ',\\,'.join('\\chgc{%s}' % c for c in e['code'])
        return '\\chgblocktag{%s}{%d}{%s}\n' % (COLOR[e['rev']], n, body) + new
    tag = inter_tag(e, n)
    if e['at'] == 'cap':
        k = new.index('\\caption{') + len('\\caption{')
        return new[:k] + tag + '\\,' + new[k:]
    if e['at'] == 'end':
        return new + tag
    return tag + ('' if new[:1].isspace() else '\\,') + new


def build_appendix(rows):
    """Apéndice A: comentarios verbatim + enlaces de vuelta + índice de cambios."""
    from review_map import REVIEWERS
    by_code = {}
    for r in rows:
        for c in r['code']:
            by_code.setdefault(c, []).append(r['n'])

    L = [r'\clearpage', r'\section*{Appendix~A\quad Reviewer comments, and where the '
         r'revision answers them}', r'\small', '',
         r'Each comment below is quoted verbatim and carries the id used by the tags in '
         r'the body text. \textbf{Answered by} lists every change that responds to it; '
         r'click a change number to jump to it. The strengths are reproduced for context '
         r'and required no edit.', '']

    for rv in REVIEWERS:
        L += ['', r'\subsection*{\textcolor{%s}{%s}}' % (rv['color'], rv['name']),
              r'\noindent{\footnotesize %s}\par\medskip' % rv['meta'], '']
        for it in rv['items']:
            nums = by_code.get(it['id'], [])
            L.append(r'\hypertarget{rc:%s}{}%%' % it['id'])
            L.append(r'\noindent\textbf{\textcolor{%s}{[%s]}}\ \textbf{%s} --- %s\par'
                     % (rv['color'], it['id'], it['kind'], it['short']))
            L.append(r'\nopagebreak')
            L.append(r"{\itshape\leftskip=1.2em\rightskip=0.6em\noindent ``%s''\par}"
                     % it['quote'])
            L.append(r'\smallskip\noindent\textbf{Response.}\ %s\par' % it['resp'])
            if nums:
                L.append(r'\noindent\textbf{Answered by:} %s.\par'
                         % ', '.join(r'\chgref{%d}' % n for n in nums))
            else:
                L.append(r'\noindent\textbf{Answered by:} no separate text change; see the '
                         r'response above.\par')
            L.append(r'\medskip')
            L.append('')
        if rv['strengths']:
            L += [r'\noindent\textbf{\textcolor{%s}{[%s-S]}}\ \textbf{Strengths} --- no '
                  r'action required.\par' % (rv['color'], rv['id']),
                  r"{\itshape\leftskip=1.2em\rightskip=0.6em\noindent ``%s''\par}"
                  % rv['strengths'], r'\medskip', '']

    L += ['', r'\subsection*{Index of changes}',
          r'\noindent{\footnotesize %d tagged changes, in reading order. The number and the '
          r'page both jump to the change in the body; the ids jump to the comment above.}'
          r'\par\smallskip'
          % len(rows),
          r'{\footnotesize', r'\setlength{\tabcolsep}{4pt}',
          r'\begin{longtable}{@{}r l l p{0.53\textwidth}@{}}',
          r'\hline', r'\# & Page & Answers & Where and what \\ \hline',
          r'\endfirsthead', r'\hline', r'\# & Page & Answers & Where and what \\ \hline',
          r'\endhead']
    for r in rows:
        ids = ',\\,'.join(r'\textcolor{%s}{\chgc{%s}}' % (COLOR[r['rev']], c)
                          for c in r['code'])
        L.append(r'\hyperlink{chg:%d}{\textcolor{chgback}{\textbf{%d}}} & '
                 r'\hyperlink{chg:%d}{\textcolor{chgback}{\pageref{chgl:%d}}} '
                 r'& %s & %s --- %s \\' % (r['n'], r['n'], r['n'], r['n'], ids,
                                           r['where'], tex_escape(r['what'])))
    L += [r'\hline', r'\end{longtable}', r'}', '']
    return '\n'.join(L)


def tex_escape(t):
    for a, b in (('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_')):
        t = t.replace(a, b)
    return t


def apply_edits(src: str, mode: str, nums=None) -> str:
    out = src
    for i, e in enumerate(EDITS):
        if e['old'] not in out:
            raise SystemExit(f"[edición {i}] ancla no encontrada ({e['note']}):\n{e['old'][:160]}")
        if out.count(e['old']) > 1:
            raise SystemExit(f"[edición {i}] ancla ambigua ({e['note']}): aparece "
                             f"{out.count(e['old'])} veces")
        if mode == 'clean':
            rep = e['new']
        elif mode == 'inter':
            rep = inter_replacement(e, nums[i])
        else:
            rep = e['tracked'] if e['tracked'] is not None else \
                default_tracked(e['rev'], e['old'], e['new'])
        out = out.replace(e['old'], rep)
    return out


def build():
    src = SRC.read_text()

    clean = apply_edits(src, 'clean')
    clean = clean.replace('% !TeX program = pdflatex\n',
                          '% !TeX program = pdflatex\n% VERSIÓN REVISADA — respuesta a los '
                          'revisores LISA 2026 (limpia). Original: paper.tex\n')
    OUT_CLEAN.write_text(clean)

    track = apply_edits(src, 'track')
    track = track.replace('% !TeX program = pdflatex\n',
                          '% !TeX program = pdflatex\n% VERSIÓN CON CONTROL DE CAMBIOS — '
                          'azul R1, naranja R2, violeta ambos, verde corrección propia.\n')
    track = track.replace('\\setlength{\\emergencystretch}{3em}',
                          '\\setlength{\\emergencystretch}{3em}\n' + PREAMBLE_TRACK)
    track = track.replace('\\maketitle\n', '\\maketitle\n' + LEGEND)
    OUT_TRACK.write_text(track)

    # La versión interactiva necesita review_map.py (texto verbatim de los
    # revisores). Ese archivo no se publica en el repositorio público, así que
    # fuera del paquete del equipo esta versión sencillamente se omite.
    if not (HERE / 'review_map.py').exists():
        print(f"{len(EDITS)} ediciones aplicadas")
        for tag in ('1', '2', 'b', '0'):
            n = sum(1 for e in EDITS if e['rev'] == tag)
            if n:
                print(f"  {TAG[tag]:6s}: {n}")
        print(f"  -> {OUT_CLEAN.name}  ({len(clean.splitlines())} líneas)")
        print(f"  -> {OUT_TRACK.name}  ({len(track.splitlines())} líneas)")
        print(f"  -- {OUT_INTER.name}: omitida (falta review_map.py)")
        return

    nums, rows = number_edits(src)
    inter = apply_edits(src, 'inter', nums)
    inter = inter.replace('% !TeX program = pdflatex\n',
                          '% !TeX program = pdflatex\n% VERSIÓN INTERACTIVA — texto limpio '
                          'con etiquetas clicables cambio <-> comentario del revisor.\n')
    inter = inter.replace('\\setlength{\\emergencystretch}{3em}',
                          '\\setlength{\\emergencystretch}{3em}\n' + PREAMBLE_TRACK
                          + PREAMBLE_INTER)
    inter = inter.replace('\\maketitle\n', '\\maketitle\n' + NAVBOX)
    inter = inter.replace('\\end{document}', build_appendix(rows) + '\n\\end{document}')
    OUT_INTER.write_text(inter)

    print(f"{len(EDITS)} ediciones aplicadas")
    for tag in ('1', '2', 'b', '0'):
        n = sum(1 for e in EDITS if e['rev'] == tag)
        if n:
            print(f"  {TAG[tag]:6s}: {n}")
    print(f"  -> {OUT_CLEAN.name}  ({len(clean.splitlines())} líneas)")
    print(f"  -> {OUT_TRACK.name}  ({len(track.splitlines())} líneas)")
    print(f"  -> {OUT_INTER.name}  ({len(inter.splitlines())} líneas, "
          f"{len(rows)} cambios etiquetados)")


def strip_marks(s: str) -> str:
    """Quita el marcado del tracked para comparar contra el revised."""
    s = s.replace(PREAMBLE_TRACK, '').replace(LEGEND, '')
    s = re.sub(r'\\rows(?:one|two|both|self)\s*', '', s)
    s = re.sub(r'\\(?:del|note)(?:one|two|both|self)\{', r'\\DELETEME{', s)
    s = drop_balanced(s, r'\DELETEME{')
    for add, _ in CMD.values():
        s = unwrap_balanced(s, '\\%s{' % add)
    s = re.sub(r'\\begin\{revblock\}\{[^}]*\}\{[^}]*\}\s*', '', s)
    s = s.replace('\\end{revblock}', '')
    return s


def _match(s, start):
    depth, i = 0, start
    while i < len(s):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError('llaves desbalanceadas')


def drop_balanced(s, opener):
    while opener in s:
        a = s.index(opener)
        b = _match(s, a + len(opener) - 1)
        s = s[:a] + s[b + 1:]
    return s


def unwrap_balanced(s, opener):
    pos = 0
    while True:
        a = s.find(opener, pos)
        if a < 0:
            return s
        b = _match(s, a + len(opener) - 1)
        s = s[:a] + s[a + len(opener):b] + s[b + 1:]
        pos = a


def norm(s):
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s+([})%,.;])', r'\1', s)
    s = re.sub(r'([{])\s+', r'\1', s)
    return s.strip()


def check():
    a = norm(strip_marks(OUT_TRACK.read_text()))
    b = norm(OUT_CLEAN.read_text())
    a = a.replace('VERSIÓN CON CONTROL DE CAMBIOS — azul R1, naranja R2, violeta ambos, verde corrección propia.', 'X')
    b = b.replace('VERSIÓN REVISADA — respuesta a los revisores LISA 2026 (limpia). Original: paper.tex', 'X')
    if a == b:
        print('OK: quitando las marcas, el tracked coincide con el revised')
        return
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f'DIFIEREN en el carácter {i}:\n  tracked: ...{a[max(0,i-120):i+120]}...\n'
                  f'  revised: ...{b[max(0,i-120):i+120]}...')
            break
    else:
        print(f'DIFIEREN en longitud: tracked {len(a)} vs revised {len(b)}')
        print('cola tracked:', a[min(len(a),len(b))-60:][:200])
        print('cola revised:', b[min(len(a),len(b))-60:][:200])
    raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    from edits import register            # noqa: E402
    register(edit)
    build()
    if args.check:
        check()
