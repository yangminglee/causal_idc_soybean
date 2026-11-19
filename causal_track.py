import os
import json
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

import idc_compute as ic
from task_nested_cv import eval_rf, _pcor_p

warnings.filterwarnings('ignore')

K_MAX = 60
M_MAX = 40
EQUAL_K = ['DPS', 'TopK_DML_P', 'TopK_DML_P_LDclump', 'TopK_L1_logistic', 'TopK_RF_importance']


def corrected_ttest(a, b, test_train_ratio=1 / 9):
    d = np.asarray(a, float) - np.asarray(b, float)
    J, var = len(d), d.var(ddof=1)
    if var == 0:
        return float(d.mean()), np.nan, (1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / np.sqrt((1 / J + test_train_ratio) * var)
    return float(d.mean()), float(t), float(2 * stats.t.sf(abs(t), J - 1))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def set_stats(S, G_tr, y, Z):
    XS = G_tr[S].values.astype(float)
    red_p = []
    for j in range(len(S)):
        Zj = np.column_stack([np.delete(XS, j, axis=1), Z]) if Z is not None else np.delete(XS, j, axis=1)
        red_p.append(_pcor_p(XS[:, [j]], y, Zj)[1][0])
    rest = [c for c in G_tr.columns if c not in set(S)]
    Zs = np.column_stack([XS, Z]) if Z is not None else XS
    _, p_rest = _pcor_p(G_tr[rest].values.astype(float), y, Zs)
    return {'size': len(S),
            'redundant': int(np.sum(np.asarray(red_p) >= 0.05)),
            'residual': int(np.sum(bh(p_rest) < 0.05))}


def one_fold(fold_dir, G, y_cont, y_bin, seed):
    r, f = map(int, os.path.basename(fold_dir)[1:].split('_f'))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed + r)
    tr, te = list(skf.split(G, y_bin))[f - 1]
    G_tr, G_te = G.iloc[tr], G.iloc[te]
    yb_tr, yb_te = y_bin.iloc[tr], y_bin.iloc[te]
    yc_tr = y_cont.iloc[tr].values.astype(float)

    dml = pd.read_csv(os.path.join(fold_dir, 'doubleml.csv'))
    order = dml['SNP'].tolist()
    p_dml = dml['P_value'].values
    sets = json.load(open(os.path.join(fold_dir, 'sets.json')))
    k_dps = len(sets['DPS'])

    _, p_raw = _pcor_p(G_tr.values.astype(float), yc_tr, np.empty((len(yc_tr), 0)))
    raw_order = G_tr.columns[np.argsort(p_raw, kind='stable')]
    n_tests = G.shape[1]
    gwas = {
        'GWAS_DML_Bonferroni': [s for s, p in zip(order, p_dml) if p < 0.05 / n_tests],
        'GWAS_DML_P1e-4': [s for s, p in zip(order, p_dml) if p < 1e-4],
        'GWAS_DML_P0.05': [s for s, p in zip(order, p_dml) if p < 0.05],
        'GWAS_raw_Bonferroni': [s for s in raw_order if p_raw[G_tr.columns.get_loc(s)] < 0.05 / n_tests],
    }

    auc = {'DPS': eval_rf(G_tr, yb_tr, G_te, yb_te, sets['DPS'], seed)['AUC']}
    curve = []
    for k in range(1, K_MAX + 1):
        curve.append({'k': k, 'AUC': eval_rf(G_tr, yb_tr, G_te, yb_te, order[:k], seed)['AUC']})
    for name, S in gwas.items():
        auc[name] = eval_rf(G_tr, yb_tr, G_te, yb_te, S, seed)['AUC'] if S else np.nan

    pcs = ic.compute_pcs(G_tr, 10, seed).values
    all_sets = {n: sets[n] for n in EQUAL_K + ['IAMB']}
    all_sets.update({n: S for n, S in gwas.items() if n != 'GWAS_DML_P0.05' and S})
    t2 = []
    for name, S in all_sets.items():
        for cond, Z in (('with_PCs', pcs), ('no_PCs', None)):
            t2.append({'Method': name, 'cond': cond, **set_stats(S, G_tr, yc_tr, Z)})

    meta = {'Repeat': r, 'Fold': f, 'k_DPS': k_dps}
    return (meta, auc, curve, t2, {n: len(S) for n, S in gwas.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--b2', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_jobs', type=int, default=30)
    ap.add_argument('--allow_dirty', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ic.write_run_info(args.out, args, args.allow_dirty)
    seed = ic.default_config()['SEED']

    G, P = ic.load_data(os.path.join(args.in_dir, 'IDC_data_for_MS2_PLOSOne.csv'))
    y_cont = P.loc[G.index, ic.TARGET_CONT].astype(float)
    y_bin = P.loc[G.index, ic.TARGET_BIN]
    dirs = sorted(glob.glob(os.path.join(args.b2, 'r?_f??')))
    assert len(dirs) == 30, len(dirs)
    res = Parallel(n_jobs=args.n_jobs)(delayed(one_fold)(d, G, y_cont, y_bin, seed) for d in dirs)

    meta = pd.DataFrame([m for m, *_ in res])
    auc = pd.DataFrame([{**m, **a} for m, a, *_ in res])
    curve = pd.DataFrame([{**m, **c} for m, _, cv, *_ in res for c in cv])
    t2 = pd.DataFrame([{**m, **row} for m, _, _, t, _ in res for row in t])
    sizes = pd.DataFrame([{**m, **s} for m, *_, s in res])
    auc.to_csv(os.path.join(args.out, 'test1_auc.csv'), index=False)
    curve.to_csv(os.path.join(args.out, 'test1_curve.csv'), index=False)
    t2.to_csv(os.path.join(args.out, 'test2_sets.csv'), index=False)
    sizes.to_csv(os.path.join(args.out, 'gwas_set_sizes.csv'), index=False)

    rep = {'n_folds': len(dirs)}
    wide = curve.pivot_table(index=['Repeat', 'Fold'], columns='k', values='AUC')
    a = auc.set_index(['Repeat', 'Fold']).loc[wide.index]
    kd = a['k_DPS'].values
    dps = a['DPS'].values
    offs = []
    for m in range(-10, M_MAX + 1):
        ks = kd + m
        if ks.min() < 1 or ks.max() > K_MAX:
            continue
        v = np.array([wide.loc[i, k] for i, k in zip(wide.index, ks)])
        d, t, p = corrected_ttest(v, dps)
        offs.append({'m': int(m), 'mean_AUC_topk': float(v.mean()), 'diff_vs_DPS': d, 't': t, 'p': p})
    offs = pd.DataFrame(offs)
    offs.to_csv(os.path.join(args.out, 'test1_offsets.csv'), index=False)
    larger = offs[offs.m >= 1]
    sig_better = larger[(larger.p < 0.05) & (larger.diff_vs_DPS > 0)]
    abs_rows = []
    for k in wide.columns:
        d, t, p = corrected_ttest(wide[k].values, dps)
        abs_rows.append({'k': int(k), 'mean_AUC': float(wide[k].mean()), 'diff_vs_DPS': d, 'p': p})
    abs_df = pd.DataFrame(abs_rows)
    not_below = abs_df[~((abs_df.p < 0.05) & (abs_df.diff_vs_DPS < 0))]
    gw = {}
    for name in ['GWAS_DML_Bonferroni', 'GWAS_DML_P1e-4', 'GWAS_DML_P0.05', 'GWAS_raw_Bonferroni']:
        v = a[name].values
        ok = ~np.isnan(v)
        d, t, p = corrected_ttest(v[ok], dps[ok])
        gw[name] = {'mean_size': float(sizes[name].mean()),
                    'size_ratio_to_DPS': float((sizes[name] / sizes['k_DPS']).mean()),
                    'mean_AUC': float(np.nanmean(v)), 'diff_vs_DPS': d, 'p': p, 'n_folds': int(ok.sum())}
    gw_fail = [n for n, g in gw.items() if g['p'] < 0.05 and g['diff_vs_DPS'] > 0]
    rep['test1'] = {
        'mean_AUC_DPS': float(dps.mean()), 'mean_k_DPS': float(kd.mean()),
        'larger_k_significantly_better': sig_better[['m', 'diff_vs_DPS', 'p']].to_dict('records'),
        'max_diff_larger_k': float(larger.diff_vs_DPS.max()),
        'smallest_k_not_significantly_below_DPS': int(not_below.k.min()),
        'gwas_cutoffs': gw,
        'C3_pass': bool(len(sig_better) == 0 and not gw_fail),
    }
    t2s = {}
    for cond in ['with_PCs', 'no_PCs']:
        sub = t2[t2.cond == cond]
        piv = {m: sub[sub.Method == m].set_index(['Repeat', 'Fold']).sort_index() for m in sub.Method.unique()}
        base = piv['DPS']
        block = {}
        for m, dfm in piv.items():
            row = {'mean_size': float(dfm['size'].mean()),
                   'mean_redundant': float(dfm['redundant'].mean()),
                   'mean_residual': float(dfm['residual'].mean())}
            if m != 'DPS':
                idx = base.index.intersection(dfm.index)
                for metric in ['redundant', 'residual']:
                    d, t, p = corrected_ttest(base.loc[idx, metric], dfm.loc[idx, metric])
                    row[f'{metric}_DPS_minus'] = d
                    row[f'{metric}_p'] = p
            block[m] = row
        t2s[cond] = block
    c = t2s['with_PCs']['TopK_DML_P']
    rep['test2'] = t2s
    rep['C4_pass'] = bool(c['redundant_DPS_minus'] < 0 and c['redundant_p'] < 0.05)
    rep['C5_pass'] = bool(c['residual_DPS_minus'] < 0 and c['residual_p'] < 0.05)
    with open(os.path.join(args.out, 'report.json'), 'w') as fh:
        json.dump(rep, fh, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    m_ = wide.mean(0)
    se = wide.std(0) / np.sqrt(len(wide))
    ax.plot(m_.index, m_.values, color='#4C72B0', label='Top-k DoubleML markers')
    ax.fill_between(m_.index, m_ - 1.96 * se, m_ + 1.96 * se, color='#4C72B0', alpha=0.2, lw=0)
    ax.axhline(dps.mean(), color='#C44E52', ls='--', lw=1, label=f'DPS (mean k = {kd.mean():.1f})')
    ax.axvline(kd.mean(), color='#C44E52', ls=':', lw=1)
    ax.set_xlabel('Number of markers k')
    ax.set_ylabel('Held-out AUC (mean of 30 folds)')
    ax.set_ylim(0.85, 1.0)
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    for ext in ('png', 'eps'):
        fig.savefig(os.path.join(args.out, f'Figure_k_curve.{ext}'), dpi=300)
    print(json.dumps({k: rep['test1'][k] for k in ['C3_pass', 'smallest_k_not_significantly_below_DPS',
                                                  'max_diff_larger_k', 'mean_AUC_DPS']}, indent=1))
    print('C4', rep['C4_pass'], 'C5', rep['C5_pass'])
    print(json.dumps(t2s['with_PCs'], indent=1))


if __name__ == '__main__':
    main()
