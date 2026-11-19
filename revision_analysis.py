import os
import json
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LassoCV, LinearRegression

warnings.filterwarnings('ignore')

SINGLE, DOUBLE, H = 3.4, 7.0, 2.5
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7, "savefig.bbox": "tight",
    "ps.fonttype": 42, "pdf.fonttype": 42,
})
Y = "IDC_score"
FULL_MODELS = ['RF', 'SVM', 'GBM', 'LR', 'rrBLUP_approx']
MODEL_LABEL = {'RF': 'RF', 'SVM': 'SVM', 'GBM': 'GBM', 'LR': 'LR', 'rrBLUP_approx': 'rrBLUP (ridge)'}
METRICS = ['Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'AUC']
C_CAUSAL, C_DPS, C_UHS, C_Y, C_OTHER = '#E15759', '#E15759', '#4E79A7', '#F1CE63', '#BAB0AC'


def savefig(fig, out, name):
    fig.savefig(os.path.join(out, name + '.eps'), dpi=600)
    fig.savefig(os.path.join(out, name + '.png'), dpi=300)
    plt.close(fig)


def corrected_ttest(a, b, test_train_ratio=1 / 9):
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float(d.mean()), np.nan, np.nan
    t = d.mean() / np.sqrt((1 / J + test_train_ratio) * var)
    return float(d.mean()), float(t), float(2 * stats.t.sf(abs(t), J - 1))


def lambda_gc(p):
    p = np.clip(np.asarray(p, float), 1e-300, 1)
    return float(stats.chi2.isf(np.median(p), 1) / stats.chi2.ppf(0.5, 1))


def pm(s, d=3):
    return f"{s.mean():.{d}f} $\\pm$ {s.std(ddof=1):.{d}f}"


def fmt_p(p):
    if p < 1e-3:
        m, e = f"{p:.2e}".split('e')
        return f"${m} \\times 10^{{{int(e)}}}$"
    return f"{p:.3f}"


def load_run(d):
    r = {'dir': d}
    r['geno'] = pd.read_csv(os.path.join(d, 'geno.csv'), index_col=0)
    r['pheno'] = pd.read_csv(os.path.join(d, 'pheno.csv'), index_col=0)
    r['pca'] = pd.read_csv(os.path.join(d, 'pca.csv'), index_col=0)
    r['dml'] = pd.read_csv(os.path.join(d, 'data_fig1_doubleml_results.csv'))
    r['t1'] = pd.read_csv(os.path.join(d, 'data_table1_key_snps.csv'))
    r['folds'] = pd.read_csv(os.path.join(d, 'data_fold_metrics.csv'))
    r['raw_graph'] = nx.read_graphml(os.path.join(d, 'data_causal_network_raw.graphml'))
    r['sets'] = json.load(open(os.path.join(d, 'feature_sets.json')))
    r['info'] = json.load(open(os.path.join(d, 'run_info.json')))
    r['dps'] = r['t1'].loc[r['t1'].Role == 'DPS', 'SNP'].tolist()
    r['uhs'] = r['t1'].loc[r['t1'].Role == 'UHS', 'SNP'].tolist()
    return r


def raw_gwas(geno, y):
    return pd.DataFrame([{'SNP': s, 'P_value': stats.linregress(geno[s].values, y).pvalue}
                         for s in geno.columns])


def fig1_and_qq(r, out, report):
    y = r['pheno'].loc[r['geno'].index, Y].values.astype(float)
    raw = raw_gwas(r['geno'], y)
    raw.to_csv(os.path.join(out, 'data_raw_gwas.csv'), index=False)
    dml = r['dml']
    stat = {}
    for name, p in [('raw', raw.P_value), ('doubleml', dml.P_value)]:
        stat[name] = {'lambda_gc': lambda_gc(p), 'n_P<0.05': int((p < 0.05).sum()),
                      'n_P<1e-4': int((p < 1e-4).sum()),
                      'n_bonferroni': int((p < 0.05 / len(p)).sum()), 'n_tests': int(len(p))}
    report['fig1'] = stat

    order = {s: i for i, s in enumerate(r['geno'].columns)}
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, H * 2), sharex=True)
    for ax, df, title, key, col in [(axes[0], raw, 'Marginal association (no adjustment)', 'raw', '#8E8E93'),
                                    (axes[1], dml, 'DoubleML (10 PCs as confounders)', 'doubleml', '#4E79A7')]:
        x = df.SNP.map(order)
        ax.scatter(x, -np.log10(df.P_value.clip(lower=1e-300)), s=2, color=col, rasterized=True)
        ax.axhline(-np.log10(0.05), color='#E15759', ls='--', lw=0.7)
        s = stat[key]
        ax.set_title(f"{title}\n$\\lambda_{{GC}}$ = {s['lambda_gc']:.2f}; P < 0.05: {s['n_P<0.05']} markers",
                     fontsize=8)
        ax.set_ylabel(r'$-\log_{10}(P)$')
        ax.spines[['top', 'right']].set_visible(False)
    axes[1].set_xlabel('Marker index (1-1,200)')
    savefig(fig, out, 'Figure1_Manhattan')

    fig, ax = plt.subplots(figsize=(SINGLE, SINGLE))
    for df, lab, col in [(raw, 'Marginal', '#8E8E93'), (dml, 'DoubleML', '#4E79A7')]:
        p = np.sort(df.P_value.clip(lower=1e-300).values)
        e = -np.log10((np.arange(1, len(p) + 1) - 0.5) / len(p))
        ax.scatter(e, -np.log10(p), s=3, color=col, label=f"{lab} ($\\lambda_{{GC}}$={lambda_gc(p):.2f})",
                   rasterized=True)
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], color='k', lw=0.7)
    ax.set_xlabel(r'Expected $-\log_{10}(P)$')
    ax.set_ylabel(r'Observed $-\log_{10}(P)$')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    savefig(fig, out, 'FigureS_QQ')


def fig_dml_diag(r, out):
    y = r['pheno'].loc[r['pca'].index, Y].values.astype(float)
    W = r['pca'].values
    m = LassoCV(cv=5, random_state=42).fit(W, y)
    resid = y - LinearRegression().fit(W, y).predict(W)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, H))
    axes[0].plot(np.log10(m.alphas_), m.mse_path_.mean(1), color='#4E79A7')
    axes[0].axvline(np.log10(m.alpha_), color='#E15759', ls='--', lw=0.7)
    axes[0].set_xlabel(r'$\log_{10}(\lambda)$ (LassoCV)')
    axes[0].set_ylabel('Cross-validated MSE')
    axes[1].hist(resid, bins=40, color='#8E8E93')
    axes[1].set_xlabel('Phenotype residual after PC adjustment')
    axes[1].set_ylabel('Count')
    for a in axes:
        a.spines[['top', 'right']].set_visible(False)
    savefig(fig, out, 'Figure2_DoubleML_diagnostics')


def fig_dag(r, out, report):
    G = r['raw_graph']
    D = nx.DiGraph([(u, v) for u, v, a in G.edges(data=True) if a.get('label') == 'd'])
    D.add_nodes_from(G.nodes())
    dps, uhs = r['dps'], r['uhs']
    inter = set()
    for h in uhs:
        try:
            inter |= set(nx.shortest_path(D, h, Y)[1:-1]) - set(dps)
        except nx.NetworkXNoPath:
            pass
    nodes = [Y] + dps + uhs + sorted(inter)
    S = D.subgraph(nodes)
    und = [(u, v) for u, v, a in G.edges(data=True) if a.get('label') == 'ud' and u in nodes and v in nodes]
    report['dag'] = {'n_nodes_shown': len(nodes), 'intermediate_nodes': sorted(inter),
                     'directed_edges_shown': S.number_of_edges(), 'undirected_edges_shown': len(und)}

    up = S.copy()
    up.remove_edges_from([(u, v) for u, v in S.edges() if u in dps and v in dps])
    layer = {Y: 0}
    for n in dps:
        layer[n] = 1
    rest = [n for n in nodes if n not in layer]
    for n in rest:
        lens = [len(p_) - 1 for t in dps for p_ in nx.all_simple_paths(up, n, t, cutoff=6)]
        layer[n] = 1 + (max(lens) if lens else 1)
    L = max(layer.values())
    rows = {k: sorted(n for n in nodes if layer[n] == k) for k in range(L + 1)}
    xs = {Y: 0.0}
    for i, n in enumerate(rows[1]):
        xs[n] = i - (len(rows[1]) - 1) / 2

    def nbr_x(n, below):
        nb = [m for m in set(S.successors(n)) | set(S.predecessors(n)) if (layer[m] < layer[n]) == below and m != Y]
        nb = [m for m in nb if m in xs]
        return np.mean([xs[m] for m in nb]) if nb else None

    def place(row, target, gap):
        order = sorted(row, key=lambda n: target[n])
        x = [target[n] for n in order]
        for _ in range(50):
            for i in range(1, len(x)):
                if x[i] - x[i - 1] < gap:
                    d = (gap - (x[i] - x[i - 1])) / 2
                    x[i - 1] -= d; x[i] += d
        return dict(zip(order, x))

    for _ in range(4):
        for k in range(2, L + 1):
            t = {n: (nbr_x(n, True) if nbr_x(n, True) is not None else xs.get(n, 0.0)) for n in rows[k]}
            xs.update(place(rows[k], t, 1.25))
        t = {}
        for n in rows[1]:
            v = nbr_x(n, False)
            t[n] = v if v is not None else xs[n]
        order = sorted(rows[1], key=lambda n: (t[n], n))
        for i, n in enumerate(order):
            xs[n] = i - (len(order) - 1) / 2
    y1 = 2.6
    pos = {n: np.array([xs[n], 0.0 if layer[n] == 0 else y1 + 1.05 * (layer[n] - 1)]) for n in nodes}

    def _clear(u, v, rad):
        a_, b_ = pos[u], pos[v]
        d = b_ - a_
        c_ = (a_ + b_) / 2 + rad * np.array([d[1], -d[0]])
        t_ = np.linspace(0.08, 0.92, 60)[:, None]
        pts = (1 - t_) ** 2 * a_ + 2 * (1 - t_) * t_ * c_ + t_ ** 2 * b_
        others = [pos[m] for m in nodes if m not in (u, v)]
        return min(np.linalg.norm(pts - o, axis=1).min() for o in others)

    def _rad(u, v):
        cands = [0.0, 0.15, -0.15, 0.3, -0.3, 0.45, -0.45, 0.6, -0.6]
        cl = {r_: _clear(u, v, r_) for r_ in cands}
        ok = [r_ for r_ in cands if cl[r_] >= 0.5]
        return ok[0] if ok else max(cands, key=lambda r_: cl[r_])

    xs_all = [p_[0] for p_ in pos.values()]
    ys_all = [p_[1] for p_ in pos.values()]
    wu, hu = max(xs_all) - min(xs_all) + 1.2, max(ys_all) - min(ys_all) + 1.0
    fig, ax = plt.subplots(figsize=(DOUBLE, DOUBLE * hu / wu + 0.5))
    for u, v in S.edges():
        to_y = v == Y
        ax.annotate('', xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle='-|>', color='#333333' if to_y else '#8E8E93',
                                    lw=1.0 if to_y else 0.8, shrinkA=10, shrinkB=10 if v != Y else 14,
                                    connectionstyle='arc3,rad=0.0' if to_y else f'arc3,rad={_rad(u, v)}'))
    for u, v in und:
        ax.plot(*zip(pos[u], pos[v]), ls=':', color='#8E8E93', lw=0.8)
    for n in nodes:
        if n == Y:
            c, sz = C_Y, 900
        elif n in dps:
            c, sz = C_DPS, 430
        elif n in uhs:
            c, sz = C_UHS, 430
        else:
            c, sz = C_OTHER, 430
        ax.scatter(*pos[n], s=sz, color=c, edgecolors='black', linewidths=0.7, zorder=3)
        if n == Y:
            ax.text(*pos[n], 'IDC', ha='center', va='center', fontsize=7.5, fontweight='bold', zorder=4)
        else:
            ax.text(*pos[n], n.replace('Marker', ''), ha='center', va='center', fontsize=5.2,
                    color='white' if n in dps or n in uhs else 'black', fontweight='bold', zorder=4)
    ax.set_xlim(min(xs_all) - 0.6, max(xs_all) + 0.6)
    ax.set_ylim(min(ys_all) - 0.5, max(ys_all) + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    report['dag']['layers'] = {str(k): v for k, v in rows.items()}
    handles = [plt.Line2D([], [], marker='o', ls='', ms=9, mfc=c, mec='k', label=l) for c, l in
               [(C_Y, 'IDC decision (target)'), (C_DPS, 'Direct parent SNP (DPS)'),
                (C_UHS, 'Upstream hub SNP (UHS)'), (C_OTHER, 'Intermediate node on a UHS path')]]
    if und:
        handles += [plt.Line2D([], [], color='#8E8E93', ls=':', label='Undirected edge (orientation not identified)')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=6.5)
    savefig(fig, out, 'Figure3_DAG')


def table1_tex(r, out):
    t = r['t1']
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"\textbf{SNP} & \textbf{Role} & \textbf{MAF} & \textbf{Effect ($\hat\theta$)} & \textbf{SE} & \textbf{$P$-value} & \textbf{Rank} \\",
             r"\midrule"]
    for _, row in t.iterrows():
        role = 'DPS' if row.Role == 'DPS' else 'UHS'
        lines.append(f"{row.SNP} & {role} & {row.MAF:.3f} & {row.Coef:.3f} & {row.SE:.3f} & {fmt_p(row.P_value)} & {int(row.DML_rank)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, 'table1_key_snps.tex'), 'w').write("\n".join(lines))


def table2(r, out, report):
    f = r['folds']
    rows, stats_rows = [], []
    crf = f[(f.Feature_Set == 'DPS') & (f.Model == 'RF')].sort_values('Fold')
    entries = [('Causal-RF (11 DPSs)' if len(r['dps']) == 11 else f"Causal-RF ({len(r['dps'])} DPSs)", crf)]
    for m in FULL_MODELS:
        entries.append((f"{MODEL_LABEL[m]} (1,200 SNPs)", f[(f.Feature_Set == 'Full') & (f.Model == m)].sort_values('Fold')))
    for name, d in entries:
        rows.append([name] + [pm(d[k]) for k in METRICS])
        if d is not crf:
            diff, t, p = corrected_ttest(crf.AUC.values, d.AUC.values)
            stats_rows.append({'Model': name, 'AUC_diff_CausalRF_minus_model': diff, 't': t, 'p_corrected': p})
    tex = [r"\begin{tabular}{lccccc}", r"\toprule",
           r"\textbf{Model (input)} & \textbf{Sensitivity} & \textbf{Specificity} & \textbf{Precision} & \textbf{Accuracy} & \textbf{AUC} \\",
           r"\midrule"] + [" & ".join(x) + r" \\" for x in rows] + [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, 'table2_decision_matrix.tex'), 'w').write("\n".join(tex))
    st = pd.DataFrame(stats_rows)
    st.to_csv(os.path.join(out, 'table2_corrected_ttests.csv'), index=False)
    report['table2_tests'] = st.round(4).to_dict('records')
    s = f.groupby(['Feature_Set', 'Model'])[METRICS].agg(['mean', 'std']).round(4)
    s.to_csv(os.path.join(out, 'tableS_all_models_all_sets.csv'))
    return entries


def fig_radar(entries, out):
    labels = METRICS
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]
    cols = [C_CAUSAL, '#4E79A7', '#F28E2B', '#59A14F', '#76B7B2', '#B07AA1']
    fig, ax = plt.subplots(figsize=(SINGLE * 1.3, SINGLE * 1.3), subplot_kw=dict(polar=True))
    lo = 0.80
    for (name, d), c in zip(entries, cols):
        v = [d[k].mean() for k in labels]
        v += v[:1]
        ax.plot(ang, v, color=c, lw=2 if c == C_CAUSAL else 1, label=name)
        if c == C_CAUSAL:
            rgb = np.array(matplotlib.colors.to_rgb(c))
            ax.fill(ang, v, color=tuple(1 - 0.12 * (1 - rgb)), zorder=0)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels)
    ax.set_ylim(lo, 1.0); ax.set_yticks([0.85, 0.90, 0.95, 1.0])
    ax.set_yticklabels(['0.85', '0.90', '0.95', '1.00'], fontsize=6)
    ax.set_rlabel_position(36)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    savefig(fig, out, 'Figure4_Radar')


def fig_box(entries, out, report):
    fig, ax = plt.subplots(figsize=(DOUBLE, H * 1.1))
    data = [d.AUC.values for _, d in entries]
    names = [n.replace(' (1,200 SNPs)', '\n(1,200 SNPs)').replace('Causal-RF ', 'Causal-RF\n') for n, _ in entries]
    b = ax.boxplot(data, patch_artist=True, widths=0.5, boxprops=dict(facecolor='#E0E0E0'),
                   medianprops=dict(color='black', lw=1.5))
    b['boxes'][0].set_facecolor('#FADBD8')
    for i, d in enumerate(data, start=1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(d)), d, s=6,
                   color='#555555', zorder=3)
        ax.text(i, max(d) + 0.004, f"median {np.median(d):.3f}", ha='center', fontsize=6.5)
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names)
    ax.set_ylabel('AUC (per fold, n = 10)')
    ax.spines[['top', 'right']].set_visible(False)
    savefig(fig, out, 'Figure5_AUC_boxplot')
    report['boxplot'] = {n: {'median': float(np.median(d.AUC)), 'sd': float(d.AUC.std(ddof=1)), 'n': int(len(d))}
                         for n, d in entries}


def fig_decay(r, out, report):
    f = r['folds']
    rf = f[f.Model == 'RF']
    order = [('DPS', f"DPSs\n(n={len(r['dps'])})"), ('UHS', f"UHSs\n(n={len(r['uhs'])})"),
             ('Random', f"Random\n(n={len(r['sets']['Random'])})"), ('Full', 'All SNPs\n(n=1,200)'),
             ('Full_minus_DPS', 'All minus DPSs\n(n=1,189)' if len(r['dps']) == 11 else 'All minus DPSs')]
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.7, H))
    x = np.arange(len(order)); w = 0.36
    res = {}
    for j, (met, col) in enumerate([('AUC', '#4E79A7'), ('Accuracy', '#F28E2B')]):
        m = [rf[rf.Feature_Set == k][met].mean() for k, _ in order]
        s = [rf[rf.Feature_Set == k][met].std(ddof=1) for k, _ in order]
        ax.bar(x + (j - 0.5) * w, m, w, yerr=s, capsize=2, color=col, label=met)
        for xi, mi, si in zip(x, m, s):
            ax.text(xi + (j - 0.5) * w, mi + si + 0.01, f"{mi:.2f}", ha='center', va='bottom', fontsize=6)
        res[met] = dict(zip([k for k, _ in order], m))
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in order])
    ax.set_ylim(0.4, 1.06); ax.set_ylabel('Score (RF, mean $\\pm$ SD)')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    savefig(fig, out, 'Figure6_SignalDecay')
    a = lambda k: rf[rf.Feature_Set == k].sort_values('Fold').AUC.values
    report['decay'] = {'mean': res,
                       'UHS_vs_Random': corrected_ttest(a('UHS'), a('Random')),
                       'FullMinusDPS_vs_Full': corrected_ttest(a('Full_minus_DPS'), a('Full'))}


def fig_kcurve(ct_dir, out, report):
    cur = pd.read_csv(os.path.join(ct_dir, 'test1_curve.csv'))
    auc = pd.read_csv(os.path.join(ct_dir, 'test1_auc.csv'))
    size = pd.read_csv(os.path.join(ct_dir, 'gwas_set_sizes.csv'))
    w = cur.pivot_table(index=['Repeat', 'Fold'], columns='k', values='AUC')
    m, se = w.mean(0), w.std(0) / np.sqrt(len(w))
    band = tuple(1 - 0.25 * (1 - np.array(matplotlib.colors.to_rgb('#4E79A7'))))
    fig, ax = plt.subplots(figsize=(SINGLE * 1.6, H * 1.15))
    ax.fill_between(m.index, m - 1.96 * se, m + 1.96 * se, color=band, lw=0, zorder=1)
    ax.plot(m.index, m.values, color='#4E79A7', lw=1.2, zorder=2, label='Top-$k$ DoubleML markers (mean, 95% CI)')
    kd, ad = auc['k_DPS'].mean(), auc['DPS'].mean()
    ax.errorbar([kd], [ad], xerr=[[kd - auc['k_DPS'].min()], [auc['k_DPS'].max() - kd]], fmt='o', ms=6,
                color=C_CAUSAL, mec='black', mew=0.6, capsize=2, zorder=4, label=f'DPSs (causal graph; $k$ = {kd:.1f})')
    marks = [('GWAS_DML_Bonferroni', 's', 'DoubleML, Bonferroni'), ('GWAS_DML_P1e-4', 'D', 'DoubleML, $P<10^{-4}$'),
             ('GWAS_raw_Bonferroni', '^', 'Unadjusted scan, Bonferroni'), ('GWAS_DML_P0.05', 'v', 'DoubleML, $P<0.05$')]
    for key, mk, lab in marks:
        ax.plot([size[key].mean()], [auc[key].mean()], mk, ms=5, color='#59A14F', mec='black', mew=0.5, zorder=3, label=lab)
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 400])
    ax.set_xticklabels(['1', '2', '5', '10', '20', '50', '100', '200', '400'])
    ax.set_xlim(0.9, 450); ax.set_ylim(0.85, 1.0)
    ax.set_xlabel('Number of markers (log scale)')
    ax.set_ylabel('Held-out AUC (nested CV, 30 folds)')
    ax.legend(frameon=False, loc='lower right', fontsize=6)
    ax.spines[['top', 'right']].set_visible(False)
    savefig(fig, out, 'Figure7_kcurve')
    report['kcurve'] = {'mean_k_DPS': float(kd), 'mean_AUC_DPS': float(ad)}


def ld_and_e1(r, out, report):
    G = r['geno'].astype(float)
    nodes = r['dps'] + r['uhs']
    r2 = G[nodes].corr() ** 2
    r2.to_csv(os.path.join(out, 'tableS_LD_r2_16nodes.csv'))
    iu = np.triu_indices(len(r['dps']), 1)
    rd = r2.loc[r['dps'], r['dps']].values[iu]
    report['ld'] = {'dps_mean_r2': float(rd.mean()), 'dps_max_r2': float(rd.max()),
                    'dps_pairs_r2_ge_0.2': int((rd >= 0.2).sum()), 'dps_pairs_r2_ge_0.5': int((rd >= 0.5).sum())}
    fig, ax = plt.subplots(figsize=(SINGLE * 1.4, SINGLE * 1.3))
    im = ax.imshow(r2.values, vmin=0, vmax=1, cmap='Reds')
    lab = [n.replace('Marker', 'M') for n in nodes]
    ax.set_xticks(range(len(nodes))); ax.set_xticklabels(lab, rotation=90, fontsize=6)
    ax.set_yticks(range(len(nodes))); ax.set_yticklabels(lab, fontsize=6)
    ax.axhline(len(r['dps']) - 0.5, color='k', lw=0.6); ax.axvline(len(r['dps']) - 0.5, color='k', lw=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, label='$r^2$')
    savefig(fig, out, 'FigureS_LD_r2')

    W = np.column_stack([np.ones(len(r['pca'])), r['pca'].loc[G.index].values])
    vc = r['dml'].loc[r['dml'].P_value < 0.05, 'SNP'].tolist()
    beta, *_ = np.linalg.lstsq(W, G[vc].values, rcond=None)
    res = G[vc].values - W @ beta
    r2pc = 1 - res.var(0) / G[vc].values.var(0)
    e1 = pd.DataFrame({'SNP': vc, 'R2_on_10PCs': r2pc})
    e1.to_csv(os.path.join(out, 'E1_R2_genotype_on_PCs.csv'), index=False)
    q = np.quantile(r2pc, [0, 0.25, 0.5, 0.75, 1])
    report['E1'] = {'n_Vcand': len(vc), 'R2_quantiles_min_q1_med_q3_max': [float(x) for x in q],
                    'frac_R2_gt_0.1': float((r2pc > 0.1).mean()),
                    'DPS_R2': {s: float(v) for s, v in zip(vc, r2pc) if s in r['dps']}}

    y = r['pheno'].loc[G.index, Y].values.astype(float)
    P = r['pca'].loc[G.index].values
    rows = []
    rank = {s: i + 1 for i, s in enumerate(r['dml'].SNP)}
    for s in r['dps']:
        o = G[[d for d in r['dps'] if d != s]].values
        for tag, Z in [('given_other_DPS', o), ('given_other_DPS_and_10PCs', np.column_stack([o, P]))]:
            Z1 = np.column_stack([np.ones(len(y)), Z])
            rx = G[s].values - Z1 @ np.linalg.lstsq(Z1, G[s].values, rcond=None)[0]
            ry = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
            pr = np.corrcoef(rx, ry)[0, 1]
            z = np.arctanh(pr) * np.sqrt(len(y) - Z.shape[1] - 3)
            rows.append({'SNP': s, 'DML_rank': rank[s], 'condition': tag, 'partial_r': pr,
                         'p': 2 * stats.norm.sf(abs(z))})
    b1 = pd.DataFrame(rows)
    b1.to_csv(os.path.join(out, 'B1_conditional_association.csv'), index=False)
    report['B1'] = b1.round(6).to_dict('records')


def e2_compare(r0, r2, report):
    a, b = set(r0['dps']), set(r2['dps'])
    f = r2['folds']
    crf = f[(f.Feature_Set == 'DPS') & (f.Model == 'RF')]
    report['E2'] = {'DPS_E0': sorted(a), 'DPS_E2': sorted(b), 'shared': sorted(a & b),
                    'only_E0': sorted(a - b), 'only_E2': sorted(b - a),
                    'UHS_E2': r2['uhs'],
                    'CausalRF_E2': {k: float(crf[k].mean()) for k in METRICS}}


def b2_summary(b2dir, out, report, dps_full):
    files = sorted(glob.glob(os.path.join(b2dir, 'r*_f*', 'metrics.csv')))
    if not files:
        return
    m = pd.concat([pd.read_csv(f) for f in files])
    m.to_csv(os.path.join(out, 'B2_all_fold_metrics.csv'), index=False)
    sets = {os.path.basename(os.path.dirname(f)): json.load(open(os.path.join(os.path.dirname(f), 'sets.json')))
            for f in files}
    timing = [json.load(open(os.path.join(os.path.dirname(f), 'timing.json'))) for f in files]
    summ = m.groupby('Method')[METRICS + ['N_features']].agg(['mean', 'std']).round(4)
    summ.to_csv(os.path.join(out, 'B2_summary.csv'))
    key = lambda d: d.sort_values(['Repeat', 'Fold']).AUC.values
    dps = m[m.Method == 'DPS']
    tests = {}
    for meth in m.Method.unique():
        if meth == 'DPS':
            continue
        tests[meth] = dict(zip(['AUC_diff_DPS_minus', 't', 'p_corrected'],
                               corrected_ttest(key(dps), key(m[m.Method == meth]))))
    methods = list(next(iter(sets.values())).keys())
    jac, freq = {}, {}
    for meth in methods:
        ss = [set(v[meth]) for v in sets.values()]
        js = [len(a & b) / len(a | b) for i, a in enumerate(ss) for b in ss[i + 1:] if a | b]
        jac[meth] = float(np.mean(js)) if js else np.nan
    allsel = pd.Series([s for v in sets.values() for s in v['DPS']]).value_counts()
    freq = {s: int(allsel.get(s, 0)) for s in dps_full}
    report['B2'] = {'n_outer_folds': len(files), 'mean_AUC': m.groupby('Method').AUC.mean().round(4).to_dict(),
                    'sd_AUC': m.groupby('Method').AUC.std().round(4).to_dict(),
                    'tests_DPS_vs_method': tests, 'mean_pairwise_jaccard': jac,
                    'selection_count_of_full_data_DPS': freq,
                    'other_frequent_DPS': allsel.head(20).to_dict(),
                    'k_per_fold': [len(v['DPS']) for v in sets.values()],
                    'timing_s_mean': {k: float(np.mean([t[k] for t in timing])) for k in ['doubleml_s', 'pc_s', 'total_s']}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e0', required=True)
    ap.add_argument('--e2')
    ap.add_argument('--b2')
    ap.add_argument('--ct')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r0 = load_run(a.e0)
    report = {'E0_run_info': r0['info'], 'E0_DPS': r0['dps'], 'E0_UHS': r0['uhs'],
              'E0_Random': r0['sets']['Random']}
    fig1_and_qq(r0, a.out, report)
    fig_dml_diag(r0, a.out)
    fig_dag(r0, a.out, report)
    table1_tex(r0, a.out)
    entries = table2(r0, a.out, report)
    fig_radar(entries, a.out)
    fig_box(entries, a.out, report)
    fig_decay(r0, a.out, report)
    ld_and_e1(r0, a.out, report)
    if a.e2:
        e2_compare(r0, load_run(a.e2), report)
    if a.b2:
        b2_summary(a.b2, a.out, report, r0['dps'])
    if a.ct:
        fig_kcurve(a.ct, a.out, report)
    with open(os.path.join(a.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != 'E0_run_info'}, indent=1, default=str))


if __name__ == "__main__":
    main()
