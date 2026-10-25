"""
R2-7 post-hoc metrics:
  1. Rule coverage per link
  2. Rule confidence, support stats
  3. Cross-client stability (antecedent overlap + consequent agreement)
  4. DQN-vs-rule action agreement (using released fuzzy_retrain results.json)
  5. Rule conflict rate (rules with same antecedent but different consequent)

Outputs a LaTeX-ready summary in `results_reviewer/rule_metrics.tex`.

Reads:
  experiments/results_StrategyB/{dataset}/afras_fl/{link}/phase5_rules.json
  experiments/results_StrategyB/{dataset}/fuzzy_retrain/results.json

Run:
  python scripts/compute_rule_metrics_all.py
"""
import json, sys, io
from pathlib import Path
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / 'experiments' / 'results_StrategyB'
OUT  = REPO / 'results_reviewer'
OUT.mkdir(exist_ok=True)

LINKS = ['Link_1_Core','Link_2_Aggregation_L','Link_3_Aggregation_R','Link_4_Edge_L','Link_5_Edge_R']
SHORT = {'Link_1_Core':'L1-Core','Link_2_Aggregation_L':'L2-AggrL','Link_3_Aggregation_R':'L3-AggrR','Link_4_Edge_L':'L4-EdgeL','Link_5_Edge_R':'L5-EdgeR'}
DATASETS = ['ciciot2023', 'diad2024']

def load_rules(ds, link):
    p = BASE / ds / 'afras_fl' / link / 'phase5_rules.json'
    return json.load(open(p)) if p.exists() else []

def summarize(ds):
    per_link = {}
    all_ante = {}
    for link in LINKS:
        rules = load_rules(ds, link)
        confs = [r['confidence'] for r in rules]
        supps = [r['support'] for r in rules]
        acts = [r['consequent'] for r in rules]
        per_link[link] = dict(
            n=len(rules),
            avg_conf=float(np.mean(confs)) if confs else 0.0,
            avg_supp=float(np.mean(supps)) if supps else 0.0,
            block=acts.count('BLOCK'), limit=acts.count('RATE_LIMIT'), allow=acts.count('ALLOW'),
        )
        for r in rules:
            key = tuple(sorted((a[0], a[1]) for a in r['antecedents']))
            all_ante.setdefault(key, []).append((link, r['consequent']))
    shared2 = sum(1 for k,v in all_ante.items() if len(v)>=2)
    shared3 = sum(1 for k,v in all_ante.items() if len(v)>=3)
    total_uniq = len(all_ante)
    consistent = sum(1 for k,v in all_ante.items() if len(v)>=2 and len(set(x[1] for x in v))==1)
    conflicting = sum(1 for k,v in all_ante.items() if len(v)>=2 and len(set(x[1] for x in v))>1)
    return per_link, dict(unique=total_uniq, shared2=shared2, shared3=shared3, consistent=consistent, conflicting=conflicting)

def dqn_agreement(ds):
    p = BASE / ds / 'fuzzy_retrain' / 'results.json'
    if not p.exists(): return {}
    res = json.load(open(p))
    out = {}
    for link in LINKS:
        if link not in res: continue
        ad = res[link]['action_distribution']
        tot = sum(ad.values())
        dqn_b = ad.get('BLOCK', 0) / tot if tot else 0
        rules = load_rules(ds, link)
        rule_b = sum(1 for r in rules if r['consequent']=='BLOCK') / len(rules) if rules else 0
        out[link] = dict(dqn_block=dqn_b, rule_block=rule_b, diff=abs(dqn_b-rule_b))
    return out

def main():
    rows = []
    for ds in DATASETS:
        per_link, stab = summarize(ds)
        agree = dqn_agreement(ds)
        print(f'\n=== {ds.upper()} ===')
        print(f'{"Client":10s} {"n":>3s} {"conf":>6s} {"supp":>10s} {"B/L/A":>7s}  {"DQN_B":>6s} {"Rule_B":>7s} {"|Δ|":>6s}')
        for link in LINKS:
            s = per_link[link]; a = agree.get(link, {})
            print(f'{SHORT[link]:10s} {s["n"]:3d} {s["avg_conf"]:6.3f} {s["avg_supp"]:10.0f} {s["block"]}/{s["limit"]}/{s["allow"]:<3d}  '
                  f'{a.get("dqn_block",0):6.3f} {a.get("rule_block",0):7.3f} {a.get("diff",0):6.3f}')
            rows.append((ds, SHORT[link], s, a))
        print(f'\nCross-client stability:  {stab["shared2"]}/{stab["unique"]} antecedents shared by >=2 clients')
        print(f'                         {stab["shared3"]}/{stab["unique"]} shared by >=3 clients')
        print(f'Consequent agreement:    {stab["consistent"]}/{stab["consistent"]+stab["conflicting"]} shared rules give identical action')
        print(f'Conflict rate:           {stab["conflicting"]}/{stab["consistent"]+stab["conflicting"]} shared rules give different actions across clients')

    # dump JSON for downstream table generation
    out = {}
    for ds in DATASETS:
        per_link, stab = summarize(ds)
        agree = dqn_agreement(ds)
        out[ds] = dict(per_link={SHORT[k]: v for k,v in per_link.items()},
                       stability=stab,
                       dqn_agreement={SHORT[k]: v for k,v in agree.items()})
    (OUT / 'rule_metrics.json').write_text(json.dumps(out, indent=2))
    print(f'\nWrote: {OUT / "rule_metrics.json"}')

if __name__ == '__main__':
    main()
