import os
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import idc_compute as ic

warnings.filterwarnings('ignore')


def ld_clump(order, G, k, r2_max=0.2):
    chosen = []
    X = G[order].values.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    idx = {s: i for i, s in enumerate(order)}
    for s in order:
        if len(chosen) == k:
            break
        if chosen:
            r = (X[:, [idx[c] for c in chosen]] * X[:, [idx[s]]]).mean(0)
            if np.max(r ** 2) >= r2_max:
                continue
        chosen.append(s)
    return chosen


def l1_topk(G, y, k, seed):
    for C in np.logspace(-4, 1, 120):
        m = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=seed).fit(G, y)
        coef = np.abs(m.coef_[0])
        if (coef > 0).sum() >= k:
            return G.columns[np.argsort(-coef, kind='stable')[:k]].tolist()
    return G.columns[np.argsort(-coef, kind='stable')[:k]].tolist()


def rf_topk(G, y, k, seed):
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1).fit(G, y)
    return G.columns[np.argsort(-rf.feature_importances_, kind='stable')[:k]].tolist()


def _resid(A, Z):
    if Z.shape[1] == 0:
        return A - A.mean(0)
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z1, A, rcond=None)
    return A - Z1 @ beta


def _pcor_p(X, y, Z):
    n = len(y)
    rx, ry = _resid(X, Z), _resid(y[:, None], Z)[:, 0]
    r = (rx * ry[:, None]).sum(0) / (np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum()) + 1e-300)
    r = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r) * np.sqrt(max(n - Z.shape[1] - 3, 1))
    return r, 2 * stats.norm.sf(np.abs(z))


def iamb(G, y, alpha=0.05, max_size=100):
    names = G.columns.tolist()
    X = G.values.astype(float)
    y = y.astype(float)
    mb = []
    while len(mb) < max_size:
        rest = [i for i in range(len(names)) if i not in mb]
        r, p = _pcor_p(X[:, rest], y, X[:, mb])
        j = int(np.argmax(np.abs(r)))
        if p[j] < alpha:
            mb.append(rest[j])
        else:
            break
    changed = True
    while changed:
        changed = False
        for i in list(mb):
            others = [m for m in mb if m != i]
            _, p = _pcor_p(X[:, [i]], y, X[:, others])
            if p[0] >= alpha:
                mb.remove(i)
                changed = True
    return sorted(names[i] for i in mb)


def eval_rf(G_tr, y_tr, G_te, y_te, feats, seed):
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(G_tr[feats], y_tr)
    y_prob = rf.predict_proba(G_te[feats])[:, 1]
    return ic.metrics(y_te, rf.predict(G_te[feats]), y_prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', default='./')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--fold', type=int, required=True, help='1..10')
    ap.add_argument('--n_jobs', type=int, default=1)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--allow_dirty', action='store_true')
    args = ap.parse_args()
    tag = f"r{args.repeat}_f{args.fold:02d}"
    out = os.path.join(args.out_dir, tag)
    os.makedirs(out, exist_ok=True)
    ic.write_run_info(out, args, args.allow_dirty)
    cfg = ic.default_config(args.debug)
    seed = cfg['SEED']
    t0 = time.time()

    G, P = ic.load_data(os.path.join(args.in_dir, "IDC_data_for_MS2_PLOSOne.csv"))
    if args.debug:
        G = G.iloc[:, :60]
    y_cont = P.loc[G.index, ic.TARGET_CONT].astype(float)
    y_bin = P.loc[G.index, ic.TARGET_BIN]

    skf = StratifiedKFold(n_splits=2 if args.debug else 10, shuffle=True, random_state=seed + args.repeat)
    tr, te = list(skf.split(G, y_bin))[args.fold - 1]
    G_tr, G_te = G.iloc[tr], G.iloc[te]
    yc_tr, yb_tr, yb_te = y_cont.iloc[tr], y_bin.iloc[tr], y_bin.iloc[te]

    pca_tr = ic.compute_pcs(G_tr, cfg['N_PCS'], seed)
    res_df, selected = ic.run_doubleml(G_tr, yc_tr, pca_tr, cfg, args.n_jobs)
    res_df.to_csv(os.path.join(out, "doubleml.csv"), index=False)
    t_dml = time.time() - t0
    data = pd.concat([G_tr[selected], yc_tr.rename(ic.TARGET_CONT)], axis=1)
    G_dir, und, bid = ic.run_pc(data, selected, cfg['PC_ALPHA'])
    dps, uhs, _ = ic.parse_roles(G_dir, cfg['TOP_HUBS_N'])
    t_pc = time.time() - t0 - t_dml
    k = max(1, len(dps))

    order = res_df['SNP'].tolist()
    sets = {
        'DPS': dps,
        'TopK_DML_P': order[:k],
        'TopK_DML_P_LDclump': ld_clump(order, G_tr, k),
        'TopK_L1_logistic': l1_topk(G_tr, yb_tr, k, seed),
        'TopK_RF_importance': rf_topk(G_tr, yb_tr, k, seed),
        'IAMB': iamb(G_tr, yc_tr.values, alpha=cfg['PC_ALPHA']),
        'UHS': uhs,
        'Random': ic.random_set(G.columns, dps + uhs, k, seed + 1000 * args.repeat + args.fold),
        'Full': G.columns.tolist(),
    }
    rows = []
    for name, feats in sets.items():
        if not feats:
            continue
        m = eval_rf(G_tr, yb_tr, G_te, yb_te, feats, seed)
        rows.append({'Repeat': args.repeat, 'Fold': args.fold, 'Method': name, 'N_features': len(feats), **m})
    pd.DataFrame(rows).to_csv(os.path.join(out, "metrics.csv"), index=False)
    with open(os.path.join(out, "sets.json"), 'w') as f:
        json.dump({k_: v for k_, v in sets.items() if k_ != 'Full'}, f, indent=2)
    with open(os.path.join(out, "timing.json"), 'w') as f:
        json.dump({'V_cand': len(selected), 'doubleml_s': t_dml, 'pc_s': t_pc,
                   'total_s': time.time() - t0, 'undirected': len(und), 'bidirected': len(bid)}, f, indent=2)
    print(pd.DataFrame(rows).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
