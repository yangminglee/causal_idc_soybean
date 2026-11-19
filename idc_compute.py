import os
import sys
import json
import argparse
import subprocess
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from doubleml import DoubleMLData, DoubleMLPLR
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

warnings.filterwarnings('ignore')

TARGET_CONT = "IDC_score"
TARGET_BIN = "IDC_class"


def default_config(debug=False):
    return {
        'DEBUG_MODE': debug,
        'N_PCS': 2 if debug else 10,
        'P_VAL_THRESHOLD': 0.99 if debug else 0.05,
        'CV_FOLDS': 2 if debug else 5,
        'PC_ALPHA': 0.05,
        'TOP_HUBS_N': 5,
        'SEED': 42,
    }


def write_run_info(out_dir, args, allow_dirty=False):
    here = os.path.dirname(os.path.abspath(__file__))
    def git(*a):
        return subprocess.run(['git', '-C', here, *a], capture_output=True, text=True).stdout.strip()
    dirty = git('status', '--porcelain', '--untracked-files=no')
    if dirty and not allow_dirty:
        sys.exit(f"Refusing to run: working tree has uncommitted changes:\n{dirty}")
    import sklearn, doubleml, causallearn, scipy
    info = {
        'commit': git('rev-parse', 'HEAD'),
        'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(dirty),
        'python': sys.version,
        'packages': {
            'numpy': np.__version__, 'pandas': pd.__version__, 'networkx': nx.__version__,
            'scikit-learn': sklearn.__version__, 'doubleml': doubleml.__version__,
            'scipy': scipy.__version__,
            'causal-learn': getattr(causallearn, '__version__', 'unknown'),
        },
        'args': vars(args),
    }
    with open(os.path.join(out_dir, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


def load_data(raw_data_file):
    df_raw = pd.read_csv(raw_data_file)
    if 'LINCD' in df_raw.columns:
        df_raw.set_index('LINCD', inplace=True)
    snp_cols = [c for c in df_raw.columns if c.startswith("Marker") or c.startswith("chr") or c.startswith("Chr")]
    pheno_cols = [c for c in ['IDC_class', 'IDC_score', 'IDC_raw', 'IDC_cat'] if c in df_raw.columns]
    return df_raw[snp_cols], df_raw[pheno_cols]


def compute_pcs(df_geno, n_pcs, seed=42):
    geno_sc = StandardScaler().fit_transform(df_geno)
    pcs = PCA(n_components=n_pcs, random_state=seed).fit_transform(geno_sc)
    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
    return pd.DataFrame(pcs, columns=pc_cols, index=df_geno.index)


def _dml_one(i, snp, df_merged, pc_cols, cv_folds, seed):
    np.random.seed(seed + i)
    ml_l = LassoCV(cv=cv_folds, random_state=seed)
    ml_m = LassoCV(cv=cv_folds, random_state=seed)
    dml_data = DoubleMLData(df_merged, y_col=TARGET_CONT, d_cols=snp, x_cols=pc_cols)
    dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=cv_folds)
    try:
        dml_plr.fit(store_predictions=False)
        s = dml_plr.summary.iloc[0]
        return {'SNP': snp, 'Coef': s['coef'], 'SE': s['std err'], 'P_value': s['P>|t|']}
    except Exception:
        return None


def run_doubleml(df_geno, y_cont, df_pca, config, n_jobs=1):
    snp_cols = df_geno.columns.tolist()
    if config['DEBUG_MODE']:
        snp_cols = snp_cols[:15]
    pc_cols = df_pca.columns.tolist()
    df_merged = pd.concat([y_cont.rename(TARGET_CONT), df_geno[snp_cols], df_pca], axis=1).dropna()
    results = Parallel(n_jobs=n_jobs)(
        delayed(_dml_one)(i, snp, df_merged, pc_cols, config['CV_FOLDS'], config['SEED'])
        for i, snp in enumerate(snp_cols))
    res_df = pd.DataFrame([r for r in results if r is not None]).sort_values('P_value').reset_index(drop=True)
    selected = res_df.loc[res_df['P_value'] < config['P_VAL_THRESHOLD'], 'SNP'].tolist()
    if config['DEBUG_MODE'] and len(selected) < 3:
        selected = res_df.head(3)['SNP'].tolist()
    return res_df, selected


def residualize(df, df_pca):
    W = np.column_stack([np.ones(len(df_pca)), df_pca.loc[df.index].values])
    beta, *_ = np.linalg.lstsq(W, df.values, rcond=None)
    return pd.DataFrame(df.values - W @ beta, index=df.index, columns=df.columns)


def cg_to_graphs(cg, names):
    G = cg.G.graph
    n = len(names)
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(names)
    und, bid = set(), set()
    for i in range(n):
        for j in range(i + 1, n):
            a, b = G[i, j], G[j, i]
            if a == 0 and b == 0:
                continue
            if a == -1 and b == 1:
                G_dir.add_edge(names[i], names[j])
            elif a == 1 and b == -1:
                G_dir.add_edge(names[j], names[i])
            elif a == -1 and b == -1:
                und.add(tuple(sorted((names[i], names[j]))))
            elif a == 1 and b == 1:
                bid.add(tuple(sorted((names[i], names[j]))))
    return G_dir, sorted(und), sorted(bid)


def drop_perfect_collinear(data_df, selected, tol=1e-10):
    X = data_df[selected].values.astype(float)
    C = np.corrcoef(X.T) if len(selected) > 1 else np.ones((1, 1))
    keep, dropped = [], []
    for j, snp in enumerate(selected):
        hit = next((selected[i] for i in keep if abs(abs(C[i, j]) - 1) < tol), None)
        if hit is None:
            keep.append(j)
        else:
            dropped.append((snp, hit))
    if dropped:
        print(f"    dropped {len(dropped)} perfectly collinear SNP(s) before PC: {dropped}")
    return [selected[i] for i in keep], dropped


def run_pc(data_df, selected, alpha):
    selected, _ = drop_perfect_collinear(data_df, selected)
    cols = selected + [TARGET_CONT]
    bk = BackgroundKnowledge()
    y_node = GraphNode(TARGET_CONT)
    for snp in selected:
        bk.add_forbidden_by_node(y_node, GraphNode(snp))
    cg = pc(data_df[cols].values, alpha=alpha, background_knowledge=bk,
            node_names=cols, verbose=False, show_progress=False)
    return cg_to_graphs(cg, cols)


def parse_roles(G_dir, top_hubs_n):
    if TARGET_CONT not in G_dir:
        return [], [], G_dir.copy()
    parents = sorted(n for n in G_dir.predecessors(TARGET_CONT) if not n.startswith('PC'))
    anc = nx.ancestors(G_dir, TARGET_CONT)
    G_sub = G_dir.subgraph(anc | {TARGET_CONT}).copy()
    hubs = [n for n in anc if n not in parents and not n.startswith('PC')]
    hubs = sorted(hubs, key=lambda n: (-G_sub.out_degree(n), n))[:top_hubs_n]
    return parents, hubs, G_sub


def quantify_linear(G, df):
    model = LinearRegression()
    for node in G.nodes():
        par = list(G.predecessors(node))
        if not par:
            continue
        model.fit(df[par], df[node])
        for p_, c in zip(par, model.coef_):
            G[p_][node]['weight'] = float(c)
    return G


def save_graph(G_dir, und, bid, path):
    H = nx.DiGraph()
    H.add_nodes_from(G_dir.nodes())
    for u, v, d in G_dir.edges(data=True):
        H.add_edge(u, v, label='d', **{k: x for k, x in d.items() if k != 'label'})
    nodes = set(G_dir.nodes())
    for u, v in und:
        if u in nodes and v in nodes:
            H.add_edge(u, v, label='ud')
    for u, v in bid:
        if u in nodes and v in nodes:
            H.add_edge(u, v, label='bd')
    nx.write_graphml(H, path)


def maf(series):
    p = (series.mean() + 1.0) / 2.0
    return float(min(p, 1 - p))


def table1(df_geno, dps, uhs, res_df):
    rank = {s: i + 1 for i, s in enumerate(res_df['SNP'])}
    rows = []
    for group, name in [(dps, 'DPS'), (uhs, 'UHS')]:
        for snp in group:
            st = res_df[res_df['SNP'] == snp]
            rows.append({'SNP': snp, 'Role': name, 'MAF': maf(df_geno[snp]),
                         'Coef': st['Coef'].values[0] if len(st) else np.nan,
                         'SE': st['SE'].values[0] if len(st) else np.nan,
                         'P_value': st['P_value'].values[0] if len(st) else np.nan,
                         'DML_rank': rank.get(snp, np.nan)})
    return pd.DataFrame(rows)


def make_models(debug=False, seed=42):
    n_est = 10 if debug else 100
    return {
        "RF": RandomForestClassifier(n_estimators=n_est, random_state=seed),
        "SVM": SVC(kernel='linear', probability=True, random_state=seed),
        "GBM": GradientBoostingClassifier(random_state=seed),
        "LR": LogisticRegression(max_iter=1000, random_state=seed),
        "rrBLUP_approx": RidgeClassifier(random_state=seed),
    }


def predict_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    s = model.decision_function(X_test)
    return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else np.zeros_like(s)


def metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_val = np.nan
    return {'AUC': auc_val, 'Accuracy': accuracy_score(y_true, y_pred),
            'Sensitivity': tp / (tp + fn) if tp + fn else np.nan,
            'Specificity': tn / (tn + fp) if tn + fp else np.nan,
            'Precision': tp / (tp + fp) if tp + fp else np.nan}


def random_set(all_snps, exclude, k, seed):
    pool = sorted(set(all_snps) - set(exclude))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist())


def run_validation(df_geno, y_bin, feature_sets, config, n_splits=10):
    skf = StratifiedKFold(n_splits=2 if config['DEBUG_MODE'] else n_splits, shuffle=True,
                          random_state=config['SEED'])
    fold_rows, prob_rows = [], []
    for fs_name, feats in feature_sets.items():
        if not feats:
            continue
        X = df_geno[feats]
        for m_name, model in make_models(config['DEBUG_MODE'], config['SEED']).items():
            for k, (tr, te) in enumerate(skf.split(X, y_bin), start=1):
                model.fit(X.iloc[tr], y_bin.iloc[tr])
                y_pred = model.predict(X.iloc[te])
                y_prob = predict_scores(model, X.iloc[te])
                fold_rows.append({'Feature_Set': fs_name, 'N_features': len(feats), 'Model': m_name,
                                  'Fold': k, **metrics(y_bin.iloc[te], y_pred, y_prob)})
                prob_rows += [{'Feature_Set': fs_name, 'Model': m_name, 'Fold': k, 'Line': idx,
                               'y_true': int(t), 'y_prob': float(p)}
                              for idx, t, p in zip(y_bin.index[te], y_bin.iloc[te], y_prob)]
    return pd.DataFrame(fold_rows), pd.DataFrame(prob_rows)


def main():
    ap = argparse.ArgumentParser(description="Soybean IDC causal pipeline (revision)")
    ap.add_argument('--in_dir', default='./')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--resid_pcs', action='store_true',
                    help='run the PC algorithm on SNPs and Y residualized on the PCs')
    ap.add_argument('--n_jobs', type=int, default=1)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--allow_dirty', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    write_run_info(args.out_dir, args, args.allow_dirty)
    cfg = default_config(args.debug)

    df_geno, df_pheno = load_data(os.path.join(args.in_dir, "IDC_data_for_MS2_PLOSOne.csv"))
    if args.debug:
        df_geno = df_geno.iloc[:, :60]
    df_geno.to_csv(os.path.join(args.out_dir, "geno.csv"))
    df_pheno.to_csv(os.path.join(args.out_dir, "pheno.csv"))
    y_cont = df_pheno.loc[df_geno.index, TARGET_CONT].astype(float)
    y_bin = df_pheno.loc[df_geno.index, TARGET_BIN]

    print(">>> [1] PCA", flush=True)
    df_pca = compute_pcs(df_geno, cfg['N_PCS'], cfg['SEED'])
    df_pca.to_csv(os.path.join(args.out_dir, "pca.csv"))

    print(">>> [2] DoubleML screening", flush=True)
    res_df, selected = run_doubleml(df_geno, y_cont, df_pca, cfg, args.n_jobs)
    res_df.to_csv(os.path.join(args.out_dir, "data_fig1_doubleml_results.csv"), index=False)
    print(f"    V_cand = {len(selected)} SNPs (P < {cfg['P_VAL_THRESHOLD']})", flush=True)

    print(f">>> [3] PC algorithm (resid_pcs={args.resid_pcs})", flush=True)
    data = pd.concat([df_geno[selected], y_cont.rename(TARGET_CONT)], axis=1)
    if args.resid_pcs:
        data = residualize(data, df_pca)
    G_dir, und, bid = run_pc(data, selected, cfg['PC_ALPHA'])
    save_graph(G_dir, und, bid, os.path.join(args.out_dir, "data_causal_network_raw.graphml"))
    dps, uhs, G_sub = parse_roles(G_dir, cfg['TOP_HUBS_N'])
    G_sub = quantify_linear(G_sub, data)
    save_graph(G_sub, und, bid, os.path.join(args.out_dir, "data_causal_network_sub.graphml"))
    print(f"    DPS ({len(dps)}): {dps}\n    UHS ({len(uhs)}): {uhs}", flush=True)
    print(f"    edges: directed={G_dir.number_of_edges()} undirected={len(und)} bidirected={len(bid)}", flush=True)

    t1 = table1(df_geno, dps, uhs, res_df)
    t1.to_csv(os.path.join(args.out_dir, "data_table1_key_snps.csv"), index=False)

    print(">>> [4] 10-fold validation", flush=True)
    all_snps = df_geno.columns.tolist()
    rnd = random_set(all_snps, dps + uhs, max(1, len(dps)), cfg['SEED'])
    feature_sets = {'Full': all_snps, 'DPS': dps, 'UHS': uhs, 'Random': rnd,
                    'Full_minus_DPS': sorted(set(all_snps) - set(dps))}
    with open(os.path.join(args.out_dir, "feature_sets.json"), 'w') as f:
        json.dump({k: v for k, v in feature_sets.items() if k not in ('Full', 'Full_minus_DPS')}, f, indent=2)
    fold_df, prob_df = run_validation(df_geno, y_bin, feature_sets, cfg)
    fold_df.to_csv(os.path.join(args.out_dir, "data_fold_metrics.csv"), index=False)
    prob_df.to_csv(os.path.join(args.out_dir, "data_roc_probs.csv"), index=False)
    summ = fold_df.groupby(['Feature_Set', 'Model'])[['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision']].agg(['mean', 'std']).round(4)
    print(summ.to_string(), flush=True)
    print(f"Done. Outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
