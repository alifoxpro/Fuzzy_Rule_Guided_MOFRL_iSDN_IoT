# -*- coding: utf-8 -*-
"""
============================================================================
One-shot integration: pull the lambda-ablation results into paper 4.

Assumes:
    paper4-Fuzzy-Augmented-Reward-Shaping/ablation/lambda_ablation.json exists
    (produced by aggregate_lambda_results.py)

Pipeline:
    1. Run compute_lambda_stats.py to produce lambda_stats.json and
       lambda_placeholders.json (t-tests, Cohen's d, etc.).
    2. Run generate_lambda_figure.py to write figures/ablation/fig_lambda_sensitivity.png.
    3. Render the three Jinja-style templates in scripts/ablation/templates/
       into concrete .tex snippets stored in ablation/rendered/.
    4. Inject:
         - the R3.1 subsection into main.tex (after the Phase comparison figure)
         - the L2 limitation text (replacing the old placeholder paragraph)
         - the sensitivity sentence into the abstract

Each injection is idempotent: running this twice will not duplicate the text.
It finds the previous injection by a pair of unique marker comments and
replaces the content between them.

Usage:
    python scripts/ablation/inject_lambda_results.py [--dry-run]

With --dry-run it prints the diff without writing to main.tex.
============================================================================
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
PAPER = PROJECT / 'paper4-Fuzzy-Augmented-Reward-Shaping'
ABL_DIR = PAPER / 'ablation'
TEMPLATES = PROJECT / 'scripts' / 'ablation' / 'templates'
MAIN_TEX = PAPER / 'main.tex'

R31_BEGIN = '% <<<LAMBDA_R31_BEGIN>>>'
R31_END = '% <<<LAMBDA_R31_END>>>'
L2_BEGIN = '% <<<LAMBDA_L2_BEGIN>>>'
L2_END = '% <<<LAMBDA_L2_END>>>'
ABS_BEGIN = '% <<<LAMBDA_ABS_BEGIN>>>'
ABS_END = '% <<<LAMBDA_ABS_END>>>'


def run_step(desc, cmd):
    print(f'\n>>> {desc}')
    print(f'    $ {" ".join(str(c) for c in cmd)}')
    r = subprocess.run(cmd, cwd=str(PROJECT))
    if r.returncode != 0:
        print(f'    FAILED with returncode {r.returncode}')
        sys.exit(r.returncode)
    print(f'    OK')


def fmt_float(x, digits=4):
    if x is None:
        return '--'
    if isinstance(x, (int, float)):
        return f'{x:.{digits}f}'
    return str(x)


def fmt_p(x):
    if x is None:
        return '--'
    if x < 1e-4:
        return f'{x:.1e}'
    return f'{x:.4f}'


def prepare_placeholders():
    """Load raw data + stats and build a flat dict of string values that
    the templates can substitute."""
    with open(ABL_DIR / 'lambda_placeholders.json') as f:
        raw = json.load(f)

    out = {}
    for k, v in raw.items():
        if k.startswith('STAT_'):
            if k.endswith('_P'):
                out[k] = fmt_p(v)
            elif k.endswith('_T') or k.endswith('_D') or k.endswith('_DIFF'):
                out[k] = fmt_float(v, 3) if v is not None else '--'
            elif k.endswith('_SDRATIO'):
                out[k] = fmt_float(v, 1) if v is not None else '--'
            else:
                out[k] = str(v)
        elif k.endswith('_MSR') or k.endswith('_MSR_STD'):
            out[k] = fmt_float(v, 4)
        elif k.endswith('_JFI'):
            out[k] = fmt_float(v, 4)
        elif k.endswith('_FPR') or k.endswith('_FNR'):
            out[k] = fmt_float(v, 4)
        elif k.endswith('_REWARD'):
            out[k] = fmt_float(v, 3)
        else:
            out[k] = str(v)

    # Derived placeholders: the MSR span (max - min) in percentage points
    def msr_span_pp(prefix):
        keys = [f'{prefix}_00_MSR', f'{prefix}_10_MSR', f'{prefix}_25_MSR', f'{prefix}_50_MSR']
        vals = [raw.get(k) for k in keys]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            return '?'
        span = (max(vals) - min(vals)) * 100
        return f'{span:.2f}'

    cic_span = msr_span_pp('LAM_CIC')
    diad_span = msr_span_pp('LAM_DIAD')
    out['CIC_MSR_SPAN_PP'] = cic_span
    out['DIAD_MSR_SPAN_PP'] = diad_span
    try:
        out['MAX_MSR_SPAN_PP'] = f'{max(float(cic_span), float(diad_span)):.2f}'
    except Exception:
        out['MAX_MSR_SPAN_PP'] = '?'

    return out


def render_template(tmpl_path, placeholders):
    txt = tmpl_path.read_text(encoding='utf-8')
    # Use a safe regex substitution so that unresolved placeholders become
    # visible (rather than crashing silently).
    def repl(m):
        key = m.group(1)
        if key in placeholders:
            return str(placeholders[key])
        print(f'    WARN: unresolved placeholder {{{{{key}}}}}')
        return '{{' + key + '}}'
    return re.sub(r'\{\{(\w+)\}\}', repl, txt)


def inject_or_replace_block(content, begin_marker, end_marker, replacement):
    """If the block exists, replace it. Otherwise return None so caller
    can handle first-time injection.

    The replacement is passed via a lambda so that backslash sequences
    in the LaTeX replacement (e.g. \\vspace, \\section) are not
    interpreted as regex backreferences.
    """
    pattern = re.compile(
        re.escape(begin_marker) + r'.*?' + re.escape(end_marker),
        re.DOTALL,
    )
    if pattern.search(content):
        full_block = begin_marker + '\n' + replacement.rstrip() + '\n' + end_marker
        return pattern.sub(lambda _m: full_block, content)
    return None


def inject_r31(content, rendered):
    wrapped = R31_BEGIN + '\n' + rendered.rstrip() + '\n' + R31_END
    replaced = inject_or_replace_block(content, R31_BEGIN, R31_END, rendered)
    if replaced is not None:
        return replaced

    # First-time injection: put it right after the Phase comparison figure
    # (fig:phase_comparison) but before the Fuzzy-TOPSIS subsection.
    anchor_re = re.compile(
        r'(\}\\label\{fig:phase_comparison\}\s*\n\\end\{figure\}\s*\n)',
    )
    m = anchor_re.search(content)
    if not m:
        print('    ERROR: could not find fig:phase_comparison anchor')
        sys.exit(1)
    insert_at = m.end()
    return content[:insert_at] + '\n' + wrapped + '\n' + content[insert_at:]


def inject_l2(content, rendered):
    wrapped = L2_BEGIN + '\n' + rendered.rstrip() + '\n' + L2_END
    replaced = inject_or_replace_block(content, L2_BEGIN, L2_END, rendered)
    if replaced is not None:
        return replaced

    # First-time: replace the old L2 paragraph (the one that says
    # "not explored" / "left to future work")
    old_l2_pattern = re.compile(
        r'\\emph\{\(L2\)[^\n]*\}[\s\S]*?'
        r'left to future work\.',
    )
    m = old_l2_pattern.search(content)
    if not m:
        print('    ERROR: could not find old L2 paragraph to replace')
        sys.exit(1)
    return content[:m.start()] + wrapped + content[m.end():]


def inject_abstract(content, rendered):
    wrapped = ABS_BEGIN + '\n' + rendered.rstrip() + '\n' + ABS_END
    replaced = inject_or_replace_block(content, ABS_BEGIN, ABS_END, rendered)
    if replaced is not None:
        return replaced

    # First-time: insert just before the closing brace of \abstract{...}.
    # We look for the specific sentence that ends the abstract in the
    # current manuscript.
    anchor_re = re.compile(
        r'(non-IID federated data\.\})',
    )
    m = anchor_re.search(content)
    if not m:
        print('    WARN: could not find abstract anchor; skipping abstract injection')
        return content
    insert_at = m.start()
    return content[:insert_at] + wrapped + '\n' + content[insert_at:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Print changes without writing to main.tex.')
    parser.add_argument('--skip-deps', action='store_true',
                        help='Skip running compute_lambda_stats and '
                             'generate_lambda_figure (use cached results).')
    args = parser.parse_args()

    # === Step 1: run dependencies ===
    if not args.skip_deps:
        run_step('Computing t-tests',
                 [sys.executable, 'scripts/ablation/compute_lambda_stats.py'])
        run_step('Generating lambda sensitivity figure',
                 [sys.executable, 'scripts/ablation/generate_lambda_figure.py'])

    # === Step 2: load placeholders ===
    placeholders = prepare_placeholders()
    print(f'\nLoaded {len(placeholders)} placeholders.')

    # === Step 3: render templates ===
    r31_rendered = render_template(
        TEMPLATES / 'r3_1_subsection.tex.tmpl', placeholders)
    l2_rendered = render_template(
        TEMPLATES / 'l2_limitation_updated.tex.tmpl', placeholders)
    abs_rendered = render_template(
        TEMPLATES / 'abstract_sensitivity_addon.tex.tmpl', placeholders)

    # Save rendered copies for inspection
    rendered_dir = ABL_DIR / 'rendered'
    rendered_dir.mkdir(parents=True, exist_ok=True)
    (rendered_dir / 'r3_1_subsection.tex').write_text(r31_rendered, encoding='utf-8')
    (rendered_dir / 'l2_limitation.tex').write_text(l2_rendered, encoding='utf-8')
    (rendered_dir / 'abstract_addon.tex').write_text(abs_rendered, encoding='utf-8')
    print(f'Wrote rendered templates to {rendered_dir}')

    # === Step 4: inject into main.tex ===
    content = MAIN_TEX.read_text(encoding='utf-8')
    original = content

    content = inject_r31(content, r31_rendered)
    print('  R3.1 section injected')

    content = inject_l2(content, l2_rendered)
    print('  L2 limitation updated')

    content = inject_abstract(content, abs_rendered)
    print('  Abstract sensitivity sentence added')

    if args.dry_run:
        print('\n=== DRY RUN: not writing main.tex ===')
        diff_summary = f'{len(content.splitlines())} vs original {len(original.splitlines())}'
        print(f'Main.tex would change from {len(original.splitlines())} to '
              f'{len(content.splitlines())} lines')
        return

    MAIN_TEX.write_text(content, encoding='utf-8')
    print(f'\nWrote {MAIN_TEX}')
    print(f'Lines changed: {len(original.splitlines())} -> {len(content.splitlines())}')

    print()
    print('Next step: recompile the paper:')
    print('    cd paper4-Fuzzy-Augmented-Reward-Shaping')
    print('    pdflatex -interaction=nonstopmode main.tex')
    print('    bibtex main')
    print('    pdflatex -interaction=nonstopmode main.tex')
    print('    pdflatex -interaction=nonstopmode main.tex')


if __name__ == '__main__':
    main()
