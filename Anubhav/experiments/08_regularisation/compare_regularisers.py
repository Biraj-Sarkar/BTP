"""Empirical answer to 'why ridge and not lasso/elastic-net/OLS'."""
import sys; sys.path.insert(0,"src")
import numpy as np, cv2
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from anemia.data import load_local_cohort
from anemia.imaging import gray_world_white_balance, read_rgb, resize
from anemia.segment import heuristic_segmenter

ROOT = Path.home()/"Desktop/BTP/Anubhav"
by = {}
for s in load_local_cohort(ROOT):
    rgb = resize(read_rgb(s.image_path), (512,512))
    bal = gray_world_white_balance(rgb)
    m,_ = heuristic_segmenter(bal)
    if not (m>0).any(): continue
    lab = cv2.cvtColor(bal, cv2.COLOR_RGB2LAB).astype(np.float32)
    sel = m>0
    feats = list(bal[sel].mean(axis=0)) + [lab[:,:,0][sel].mean(),
             lab[:,:,1][sel].mean()-128, lab[:,:,2][sel].mean()-128]
    by.setdefault(s.patient_id, {"hb": s.hb, "f": []})["f"].append(np.array(feats))

pts = sorted(by)
X = np.array([np.mean(by[p]["f"],axis=0) for p in pts]); y = np.array([by[p]["hb"] for p in pts])
print(f"{len(pts)} patients, {X.shape[1]} features (R,G,B,L*,a*,b* means)\n")

# how correlated are the features? this is the crux of ridge-vs-lasso
C = np.corrcoef(X.T)
off = C[~np.eye(len(C),dtype=bool)]
print(f"feature correlation: max |r| = {np.abs(off).max():.3f}, mean |r| = {np.abs(off).mean():.3f}")
print("(highly correlated predictors are exactly where lasso becomes unstable)\n")

models = [("OLS (no penalty)", LinearRegression()),
          ("Ridge  alpha=1", Ridge(alpha=1.0)),
          ("Ridge  alpha=10", Ridge(alpha=10.0)),
          ("Lasso  alpha=0.1", Lasso(alpha=0.1, max_iter=50000)),
          ("Lasso  alpha=0.5", Lasso(alpha=0.5, max_iter=50000)),
          ("ElasticNet 0.5", ElasticNet(alpha=0.3, l1_ratio=0.5, max_iter=50000))]

print(f"{'model':<20}{'LOO MAE':>9}{'features kept':>15}{'coef stability':>16}")
print("-"*62)
base = np.mean([abs(np.delete(y,i).mean()-y[i]) for i in range(len(y))])
for name, mdl in models:
    errs, coefs, kept = [], [], []
    for i in range(len(y)):
        k = np.ones(len(y),bool); k[i]=False
        mu,sd = X[k].mean(0), X[k].std(0); sd[sd<1e-9]=1
        mdl.fit((X[k]-mu)/sd, y[k])
        errs.append(abs(mdl.predict(((X[i]-mu)/sd).reshape(1,-1))[0]-y[i]))
        coefs.append(mdl.coef_.copy()); kept.append(int(np.count_nonzero(np.abs(mdl.coef_)>1e-8)))
    coefs=np.array(coefs)
    # stability: how much each coefficient wobbles across the 26 refits
    stab = float(np.mean(np.std(coefs,axis=0)))
    print(f"{name:<20}{np.mean(errs):>9.3f}{np.mean(kept):>15.1f}{stab:>16.3f}")
print("-"*62)
print(f"{'baseline (mean)':<20}{base:>9.3f}")
print("\n'features kept' = non-zero coefficients (lasso zeroes some out)")
print("'coef stability' = mean SD of each coefficient across the 26 refits; lower = steadier")
