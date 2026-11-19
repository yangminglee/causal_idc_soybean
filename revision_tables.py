import os
import json
import argparse

import numpy as np
import pandas as pd

METHODS = [
    ('DPS', 'Causal (DPSs, PC algorithm)'),
    ('TopK_DML_P', 'Top-$k$ DoubleML $P$'),
    ('TopK_DML_P_LDclump', 'Top-$k$ DoubleML $P$, LD-clumped'),
    ('TopK_L1_logistic', 'Top-$k$ L1-logistic'),
    ('TopK_RF_importance', 'Top-$k$ RF importance'),
    ('IAMB', 'IAMB Markov blanket'),
    ('UHS', 'UHSs'),
    ('Random', 'Random $k$ markers'),
    ('Full', 'All 1,200 markers'),
]


def fp(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '--'
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return f'${p / 10 ** e:.1f}\\times10^{{{e}}}$'
    return f'{p:.2f}' if p >= 0.1 else f'{p:.3f}'


def pm(m, s, d=3):
    return f'{m:.{d}f} $\\pm$ {s:.{d}f}'


def nested_table(a_dir, ct_dir, out):
    rep = json.load(open(os.path.join(a_dir, 'report.json')))['B2']
    f = pd.read_csv(os.path.join(a_dir, 'B2_all_fold_metrics.csv'))
    t2 = pd.read_csv(os.path.join(ct_dir, 'test2_sets.csv'))
    ctr = json.load(open(os.path.join(ct_dir, 'report.json')))['test2']['with_PCs']
    t2 = t2[t2.cond == 'with_PCs'].groupby('Method')[['redundant', 'residual']].mean()
    g = f.groupby('Method')
    rows = []
    for key, label in METHODS:
        d = g.get_group(key)
        test = rep['tests_DPS_vs_method'].get(key)
        jac = rep['mean_pairwise_jaccard'].get(key)
        red = t2['redundant'].get(key, np.nan)
        res = t2['residual'].get(key, np.nan)
        p_res = ctr.get(key, {}).get('residual_p') if key != 'DPS' else None
        rows.append(' & '.join([
            label,
            '1,200' if key == 'Full' else f"{d['N_features'].mean():.1f}",
            pm(d['Sensitivity'].mean(), d['Sensitivity'].std()),
            pm(d['Specificity'].mean(), d['Specificity'].std()),
            pm(d['AUC'].mean(), d['AUC'].std()),
            '--' if test is None else fp(test['p_corrected']),
            '--' if jac is None else f'{jac:.2f}',
            '--' if np.isnan(red) else f'{red:.1f}',
            '--' if np.isnan(res) else f'{res:.1f}',
            '--' if p_res is None else fp(p_res),
        ]) + r' \\')
    head = (r'\begin{tabular}{lccccccccc}' '\n' r'\toprule' '\n'
            r'\textbf{Marker set} & \textbf{Size} & \textbf{Sensitivity} & \textbf{Specificity} & '
            r'\textbf{AUC} & \textbf{$P_{\mathrm{AUC}}$} & \textbf{Jaccard} & '
            r'\textbf{Redundant} & \textbf{Residual} & \textbf{$P_{\mathrm{res}}$} \\' '\n' r'\midrule')
    open(os.path.join(out, 'nested_cv.tex'), 'w').write(head + '\n' + '\n'.join(rows) + '\n'
                                                         r'\bottomrule' '\n' r'\end{tabular}' '\n')


def gwas_table(ct_dir, out):
    t1 = json.load(open(os.path.join(ct_dir, 'report.json')))['test1']
    lab = {
        'GWAS_DML_Bonferroni': r'DoubleML, $P < 0.05/1{,}200$ (Bonferroni)',
        'GWAS_DML_P1e-4': r'DoubleML, $P < 10^{-4}$',
        'GWAS_DML_P0.05': r'DoubleML, $P < 0.05$ ($V_{cand}$)',
        'GWAS_raw_Bonferroni': r'Unadjusted marginal scan, Bonferroni',
    }
    rows = [' & '.join(['Causal (DPSs)', f"{t1['mean_k_DPS']:.1f}", '1.0', f"{t1['mean_AUC_DPS']:.3f}", '--', '--']) + r' \\']
    for k, l in lab.items():
        g = t1['gwas_cutoffs'][k]
        rows.append(' & '.join([l, f"{g['mean_size']:.1f}", f"{g['size_ratio_to_DPS']:.1f}", f"{g['mean_AUC']:.3f}",
                                f"{g['diff_vs_DPS']:+.3f}", fp(g['p'])]) + r' \\')
    head = (r'\begin{tabular}{lccccc}' '\n' r'\toprule' '\n'
            r'\textbf{Marker set} & \textbf{Size} & \textbf{Size / $|$DPS$|$} & \textbf{AUC} & '
            r'\textbf{$\Delta$AUC vs DPS} & \textbf{$P$} \\' '\n' r'\midrule')
    open(os.path.join(out, 'gwas_cutoffs.tex'), 'w').write(head + '\n' + '\n'.join(rows) + '\n'
                                                            r'\bottomrule' '\n' r'\end{tabular}' '\n')


def supp_tables(a_dir, ct_dir, out):
    rep = json.load(open(os.path.join(a_dir, 'report.json')))
    ld = pd.read_csv(os.path.join(a_dir, 'tableS_LD_r2_16nodes.csv'), index_col=0)
    names = [c.replace('Marker', '') for c in ld.columns]
    lines = [r'\begin{tabular}{l' + 'c' * len(names) + '}', r'\toprule',
             ' & '.join([''] + [f'\\rotatebox{{90}}{{{n}}}' for n in names]) + r' \\', r'\midrule']
    for i, r_ in enumerate(ld.index):
        cells = [r_.replace('Marker', '')]
        for j in range(len(names)):
            cells.append('' if j >= i else f'{ld.iloc[i, j]:.2f}')
        lines.append(' & '.join(cells) + r' \\')
        if i == 10:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'S1_ld.tex'), 'w').write('\n'.join(lines) + '\n')
    m = ld.values[np.triu_indices(len(ld), 1)]
    dps_m = ld.values[:11, :11][np.triu_indices(11, 1)]
    json.dump({'all16_mean': float(m.mean()), 'all16_max': float(m.max()), 'all16_n_ge_0.2': int((m >= 0.2).sum()),
               'all16_pairs': int(len(m)), 'dps_mean': float(dps_m.mean()), 'dps_max': float(dps_m.max()),
               'dps_n_ge_0.2': int((dps_m >= 0.2).sum()), 'dps_pairs': int(len(dps_m))},
              open(os.path.join(out, 'S1_ld_summary.json'), 'w'), indent=1)

    b1 = pd.read_csv(os.path.join(a_dir, 'B1_conditional_association.csv'))
    t1 = pd.read_csv(os.path.join(a_dir, 'report.json').replace('report.json', 'B1_conditional_association.csv'))
    sel = rep['B2']['selection_count_of_full_data_DPS']
    n = rep['B2']['n_outer_folds']
    lines = [r'\begin{tabular}{lccccc}', r'\toprule',
             r'\textbf{DPS} & \textbf{DoubleML rank} & \textbf{Partial $r$ $\mid$ DPSs} & \textbf{$P$} & '
             r'\textbf{Partial $r$ $\mid$ DPSs + 10 PCs} & \textbf{$P$} \\', r'\midrule']
    for snp in sorted(b1.SNP.unique(), key=lambda s: b1[b1.SNP == s].DML_rank.iloc[0]):
        a = b1[(b1.SNP == snp) & (b1.condition == 'given_other_DPS')].iloc[0]
        b = b1[(b1.SNP == snp) & (b1.condition == 'given_other_DPS_and_10PCs')].iloc[0]
        lines.append(' & '.join([snp, str(int(a.DML_rank)), f'{a.partial_r:.3f}', fp(a.p), f'{b.partial_r:.3f}', fp(b.p)]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'S2_conditional.tex'), 'w').write('\n'.join(lines) + '\n')

    lines = [r'\begin{tabular}{lcc}', r'\toprule',
             r'\textbf{Marker} & \textbf{Role in full-data graph} & \textbf{Selected as DPS (of ' + str(n) + r' training sets)} \\', r'\midrule']
    other = rep['B2']['other_frequent_DPS']
    for snp, c in sorted(other.items(), key=lambda kv: (-kv[1], kv[0])):
        if c < 2 and snp not in sel:
            continue
        lines.append(f"{snp} & {'DPS' if snp in sel else 'not a DPS'} & {c} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'S3_selection.tex'), 'w').write('\n'.join(lines) + '\n')

    e2 = rep['E2']
    lines = [r'\begin{tabular}{lp{9cm}}', r'\toprule', r'\textbf{Item} & \textbf{Result} \\', r'\midrule',
             f"DPSs shared with the main analysis & {len(e2['shared'])}: " + ', '.join(s.replace('Marker', '') for s in e2['shared']) + r' \\',
             f"DPSs only in the main analysis & {len(e2['only_E0'])}: " + ', '.join(s.replace('Marker', '') for s in e2['only_E0']) + r' \\',
             f"DPSs only after PC residualization & {len(e2['only_E2'])}: " + ', '.join(s.replace('Marker', '') for s in e2['only_E2']) + r' \\',
             'Causal-RF on the residualization DPSs & ' + ', '.join(f"{k} {v:.3f}" for k, v in e2['CausalRF_E2'].items()) + r' \\',
             r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'S4_e2.tex'), 'w').write('\n'.join(lines) + '\n')

    t2 = pd.read_csv(os.path.join(ct_dir, 'test2_sets.csv'))
    g = t2.groupby(['cond', 'Method'])[['size', 'redundant', 'residual']].mean()
    order = ['DPS', 'TopK_DML_P', 'TopK_DML_P_LDclump', 'TopK_L1_logistic', 'TopK_RF_importance', 'IAMB',
             'GWAS_DML_Bonferroni', 'GWAS_DML_P1e-4', 'GWAS_raw_Bonferroni']
    lab = dict(METHODS)
    lab.update({'GWAS_DML_Bonferroni': 'DoubleML, Bonferroni', 'GWAS_DML_P1e-4': r'DoubleML, $P<10^{-4}$',
                'GWAS_raw_Bonferroni': 'Unadjusted scan, Bonferroni'})
    lines = [r'\begin{tabular}{lccccc}', r'\toprule',
             r' & & \multicolumn{2}{c}{\textbf{Conditioning on 10 PCs}} & \multicolumn{2}{c}{\textbf{No PCs}} \\',
             r'\textbf{Marker set} & \textbf{Size} & \textbf{Redundant} & \textbf{Residual} & \textbf{Redundant} & \textbf{Residual} \\',
             r'\midrule']
    for m_ in order:
        a, b = g.loc[('with_PCs', m_)], g.loc[('no_PCs', m_)]
        lines.append(' & '.join([lab[m_], f'{a["size"]:.1f}', f'{a.redundant:.1f}', f'{a.residual:.1f}',
                                 f'{b.redundant:.1f}', f'{b.residual:.1f}']) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'S5_test2.tex'), 'w').write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', required=True)
    ap.add_argument('--ct', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nested_table(a.analysis, a.ct, a.out)
    gwas_table(a.ct, a.out)
    supp_tables(a.analysis, a.ct, a.out)


if __name__ == '__main__':
    main()
