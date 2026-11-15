"""Compare short runs on val: RMSE table, % vs A0, paired bootstrap CIs over starts.
Usage: python scripts/compare_runs.py <results_dir> [persist|mask|oracle]"""
import pandas as pd, numpy as np, json
import sys
R = sys.argv[1] if len(sys.argv) > 1 else "results"   # folder with <run>/eval_*/metrics_raw.csv
POL = sys.argv[2] if len(sys.argv) > 2 else 'persist'
runs = ['v2_short','abl_no_time','abl_no_sat','abl_legacy_edges','f1_concat_sub','f2_gated','f3_xattn','v3_rollout4','v3_rollout12']
import os
runs = [r for r in runs if os.path.exists(f'{R}/{r}/eval_{POL}/metrics_raw.csv')]
raw, cfgs = {}, {}
for r in runs:
    d = pd.read_csv(f"{R}/{r}/eval_{POL}/metrics_raw.csv")
    raw[r] = d[(d.region=="all")]
    cfgs[r] = json.load(open(f"{R}/{r}/eval_{POL}/eval_config.json"))
mname = {r: [m for m in raw[r].model.unique() if m not in ("persistence","climatology")][0] for r in runs}
print({r:(cfgs[r]["n_params"], cfgs[r]["git_commit"][:7]) for r in runs})
CH = ["t2m","u10","v10","msl","z500","q850","tp"]
def mean_rmse(r, ch, L):
    d = raw[r]; return d[(d.model==mname[r])&(d.channel==ch)&(d.lead_h==L)].rmse.mean()
rows=[]
for r in runs:
    row={"run":r,"params":cfgs[r]["n_params"]}
    for ch in CH:
        for L in (24,72):
            row[f"{ch}@{L}"]=mean_rmse(r,ch,L)
    rows.append(row)
T = pd.DataFrame(rows).set_index("run")
base = T.loc["v2_short"]
P = (T.drop(columns="params") / base.drop("params") - 1) * 100
pd.set_option("display.width", 250)
print("\n=== RMSE (val 2025, all region, mean of 30 starts)"); print(T.round(4).to_string())
print("\n=== % change vs A0 (v2_short); negative = better"); print(P.round(1).to_string())
# composite: mean % over target vars t2m,u10,v10 over all leads 6..72
def comp(r, chs=("t2m","u10","v10")):
    out=[]
    for ch in chs:
        for L in range(6,73,6):
            out.append(mean_rmse(r,ch,L)/mean_rmse("v2_short",ch,L)-1)
    return 100*np.mean(out)
print("\n=== composite Δ% (t2m,u10,v10, all leads) vs A0:", {r: round(comp(r),2) for r in runs})
print("=== composite Δ% (all 7 vars):", {r: round(comp(r,CH),2) for r in runs})
# paired per-start: fraction of starts where run beats A0, + bootstrap 95% CI of mean diff (%), target vars at 24 & 72
rng = np.random.default_rng(0)
def paired(r, ref="v2_short", chs=("t2m","u10","v10"), leads=(24,48,72)):
    res={}
    a = raw[r][raw[r].model==mname[r]]; b = raw[ref][raw[ref].model==mname[ref]]
    m = a.merge(b, on=["start","lead_h","channel"], suffixes=("_r","_b"))
    m = m[m.channel.isin(chs) & m.lead_h.isin(leads)]
    m["rel"] = m.rmse_r/m.rmse_b - 1
    per_start = m.groupby("start").rel.mean()
    boots = [per_start.sample(len(per_start), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
    return round(100*per_start.mean(),2), (round(100*np.percentile(boots,2.5),2), round(100*np.percentile(boots,97.5),2)), f"{(per_start<0).sum()}/{len(per_start)}"
print("\n=== paired vs A0 over 30 starts (t2m,u10,v10 @24/48/72): mean Δ%, 95% CI, #starts better")
for r in runs[1:7]: print(f"  {r:18s}", paired(r))
print("\n=== F2/F3 vs F1 (same satellite info):")
for r in ("f2_gated","f3_xattn"): print(f"  {r:18s}", paired(r, ref="f1_concat_sub"))
print("  f3 vs f2          ", paired("f3_xattn", ref="f2_gated"))
print("\n=== v3b vs v3a (all 7 vars @48/72):", paired("v3_rollout12", ref="v3_rollout4", chs=CH, leads=(48,72)))
print("    v3b vs v3a (t2m,u10,v10 @6/24):", paired("v3_rollout12", ref="v3_rollout4", leads=(6,24)))
T.to_csv("rmse_table.csv"); P.to_csv("pct_vs_A0.csv")

print("\n=== satellite effect by lead (paired vs abl_no_sat = no satellite), t2m,u10,v10,tp; negative = satellite helps")
for r in ("v2_short","f1_concat_sub","f2_gated","f3_xattn"):
    out = []
    for L in (6,12,24,48,72):
        out.append((L, paired(r, ref="abl_no_sat", chs=("t2m","u10","v10","tp"), leads=(L,))[0]))
    print(f"  {r:16s}", out)
print("\n=== time/solar effect by start hour, t2m @6..24 (abl_no_time vs A0, + = time features help)")
a = raw["abl_no_time"]; b = raw["v2_short"]
m = a[(a.model==mname["abl_no_time"])].merge(b[b.model==mname["v2_short"]], on=["start","lead_h","channel"], suffixes=("_nt","_a0"))
m = m[(m.channel=="t2m")&(m.lead_h<=24)]
m["hr"] = m.start.str[11:13]
print((100*(m.rmse_nt/m.rmse_a0-1)).groupby([m.hr, m.lead_h]).mean().unstack().round(1).to_string())
