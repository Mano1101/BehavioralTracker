"""Design-aware advanced statistics for BehavioralTracker.

The module deliberately separates descriptive structure from inferential models.
It never invents Group/Animal/Dam/Cage relationships. Repeated observations are
modelled with subject-aware methods when enough metadata are present.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SUBJECT_CANDIDATES = ["Animal ID","Animal_ID","Animal","animal_id","animal","Subject","Subject_ID","subject_id","subject"]
GROUP_CANDIDATES = ["Group","group"]
TIME_CANDIDATES = ["Timepoint","timepoint","Session","session"]
DAM_CANDIDATES = ["Dam ID","Dam/Cage","Dam_Cage","Dam","dam","dam_id"]
CAGE_CANDIDATES = ["Cage ID","Cage","cage","cage_id"]


def _first(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def _numeric_metrics(df, exclude):
    out=[]
    for c in df.columns:
        if c in exclude: continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() >= 2:
            out.append(c)
    return out


def _cohen_d_ci(a,b, bootstrap=2000, seed=1234):
    a=np.asarray(a,float); b=np.asarray(b,float)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<2 or len(b)<2: return np.nan,np.nan,np.nan
    pooled=np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
    d=(np.mean(a)-np.mean(b))/pooled if pooled>0 else np.nan
    if not np.isfinite(d): return d,np.nan,np.nan
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(bootstrap):
        aa=rng.choice(a,len(a),replace=True); bb=rng.choice(b,len(b),replace=True)
        pp=np.sqrt(((len(aa)-1)*np.var(aa,ddof=1)+(len(bb)-1)*np.var(bb,ddof=1))/(len(aa)+len(bb)-2))
        vals.append((aa.mean()-bb.mean())/pp if pp>0 else np.nan)
    vals=np.asarray(vals); vals=vals[np.isfinite(vals)]
    return d,float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5)) if len(vals) else np.nan


def design_audit(df):
    subject=_first(df,SUBJECT_CANDIDATES); group=_first(df,GROUP_CANDIDATES); time=_first(df,TIME_CANDIDATES); dam=_first(df,DAM_CANDIDATES); cage=_first(df,CAGE_CANDIDATES)
    rows=[]
    for label,col in [("Animal/Subject",subject),("Group",group),("Timepoint/Session",time),("Dam",dam),("Cage",cage)]:
        rows.append({"Field":label,"Column":col or "MISSING","Present":"yes" if col else "no","Unique":int(df[col].nunique(dropna=True)) if col else 0,"Missing":int(df[col].isna().sum()) if col else len(df)})
    if subject:
        counts=df[subject].astype(str).value_counts()
        rows.append({"Field":"Repeated subjects","Column":subject,"Present":"yes","Unique":int((counts>1).sum()),"Missing":0})
    return pd.DataFrame(rows)


def subject_level_summary(df, metrics=None):
    subject=_first(df,SUBJECT_CANDIDATES); group=_first(df,GROUP_CANDIDATES); time=_first(df,TIME_CANDIDATES); dam=_first(df,DAM_CANDIDATES); cage=_first(df,CAGE_CANDIDATES)
    if not subject: raise ValueError("Animal/Subject ID is required for subject-level analysis.")
    metrics=metrics or _numeric_metrics(df,{subject,group,time,dam,cage})
    keys=[subject]+([group] if group else [])+([dam] if dam else [])+([cage] if cage else [])
    if time: keys.append(time)
    work=df.copy(); work[subject]=work[subject].astype(str)
    return work.groupby(keys,dropna=False)[metrics].mean().reset_index()


def repeated_measures(df, metric_cols=None):
    subject=_first(df,SUBJECT_CANDIDATES); group=_first(df,GROUP_CANDIDATES); time=_first(df,TIME_CANDIDATES)
    if not subject or not time: return pd.DataFrame()
    metrics=metric_cols or _numeric_metrics(df,{subject,group,time,_first(df,DAM_CANDIDATES)})
    s=subject_level_summary(df,metrics)
    levels=list(s[time].dropna().astype(str).unique())
    rows=[]
    if len(levels)<2: return pd.DataFrame()
    # For two timepoints, compute paired within-subject change. This is a descriptive/inferential
    # repeated-measures result; it does not pretend missing pairs are independent observations.
    for metric in metrics:
        pvt=s.pivot_table(index=subject,columns=time,values=metric,aggfunc="mean")
        if len(levels)==2:
            a,b=levels[0],levels[1]
            pair=pvt[[a,b]].dropna() if a in pvt and b in pvt else pd.DataFrame()
            if len(pair)>=2:
                diff=pair[b]-pair[a]
                try: stat,p=stats.ttest_rel(pair[b],pair[a]); test="Paired t-test"
                except Exception: stat,p=np.nan,np.nan; test="insufficient data"
                if not np.isfinite(p) or len(pair)<3:
                    try: stat,p=stats.wilcoxon(pair[b],pair[a]); test="Wilcoxon signed-rank"
                    except Exception: pass
                se=diff.std(ddof=1)/np.sqrt(len(diff)) if len(diff)>1 else np.nan
                ci_low=diff.mean()-stats.t.ppf(.975,len(diff)-1)*se if len(diff)>1 else np.nan
                ci_high=diff.mean()+stats.t.ppf(.975,len(diff)-1)*se if len(diff)>1 else np.nan
                rows.append({"Metric":metric,"Time_A":a,"Time_B":b,"N_pairs":len(pair),"Mean_change":float(diff.mean()),"CI95_low":ci_low,"CI95_high":ci_high,"Test":test,"Statistic":float(stat) if np.isfinite(stat) else np.nan,"p_value":float(p) if np.isfinite(p) else np.nan})
        else:
            # Global within-subject omnibus only when complete cases exist across all timepoints.
            pair=pvt.dropna(subset=levels)
            if len(pair)>=2:
                try: stat,p=stats.friedmanchisquare(*[pair[x].to_numpy() for x in levels]); test="Friedman test"
                except Exception: stat,p=np.nan,np.nan; test="insufficient data"
                rows.append({"Metric":metric,"Time_A":"all","Time_B":"all","N_pairs":len(pair),"Mean_change":np.nan,"CI95_low":np.nan,"CI95_high":np.nan,"Test":test,"Statistic":float(stat) if np.isfinite(stat) else np.nan,"p_value":float(p) if np.isfinite(p) else np.nan})
    out=pd.DataFrame(rows)
    if "p_value" in out.columns and out["p_value"].notna().any():
        from analysis.statistics import _bh_adjust
        out["p_adjusted_BH"]=_bh_adjust(out["p_value"].to_numpy(dtype=float))
        out["Correction"]=f"Benjamini–Hochberg FDR over the {int(out['p_value'].notna().sum())} metrics"
    return out


def mixed_effects(df, metric_cols=None):
    """Fit subject-random-intercept models with optional explicit Dam→Cage nesting."""
    subject=_first(df,SUBJECT_CANDIDATES); group=_first(df,GROUP_CANDIDATES); time=_first(df,TIME_CANDIDATES); dam=_first(df,DAM_CANDIDATES); cage=_first(df,CAGE_CANDIDATES)
    if not subject: return pd.DataFrame([{"Status":"Animal/Subject ID required"}])
    metrics=metric_cols or _numeric_metrics(df,{subject,group,time,dam,cage})
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return pd.DataFrame([{"Status":"statsmodels is not installed; mixed-effects analysis unavailable."}])
    rows=[]
    for metric in metrics:
        cols=[subject,metric]+([group] if group else [])+([time] if time else [])+([dam] if dam else [])+([cage] if cage else [])
        d=df[cols].dropna().copy()
        if len(d)<4: rows.append({"Metric":metric,"Status":"insufficient rows"}); continue
        # Use safe internal names so arbitrary user column names cannot break Patsy formulas.
        rename={subject:"__subject", metric:"__metric"}
        if group: rename[group]="__group"
        if time: rename[time]="__time"
        if dam: rename[dam]="__dam"
        if cage: rename[cage]="__cage"
        d=d.rename(columns=rename); d["__subject"]=d["__subject"].astype(str)
        terms=[]
        if group: terms.append("C(__group)")
        if time: terms.append("C(__time)")
        if group and time: terms.append("C(__group):C(__time)")
        if not terms: rows.append({"Metric":metric,"Status":"Need Group and/or Timepoint"}); continue
        formula="__metric ~ " + " + ".join(terms)
        try:
            vc = None
            if dam and cage and d["__dam"].nunique()>1 and d["__cage"].nunique()>1:
                d["__dam_cage"] = d["__dam"].astype(str) + "::" + d["__cage"].astype(str)
                vc = {"Dam_Cage": "0 + C(__dam_cage)"}
            elif dam and d["__dam"].nunique()>1:
                vc = {"Dam": "0 + C(__dam)"}
            import warnings
            estimation="maximum likelihood (REML off), L-BFGS"
            kind="mixed"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model=smf.mixedlm(formula,d,groups=d["__subject"],vc_formula=vc,re_formula="1")
                    fit=model.fit(reml=False,method="lbfgs",disp=False)
                    ci=fit.conf_int()
                    dam_status="Dam→Cage nested variance component included" if (vc and cage and dam) else ("Dam variance component included" if vc else "not modeled")
                except Exception:
                    # A sparse/over-parameterised Dam/Cage variance component can be singular.
                    # Retry with the subject random intercept only and report the limitation.
                    dam_status="not modeled (singular Dam/Cage variance component)" if vc else "not modeled"
                    try:
                        model=smf.mixedlm(formula,d,groups=d["__subject"],re_formula="1")
                        fit=model.fit(reml=False,method="lbfgs",disp=False)
                        ci=fit.conf_int()
                    except Exception:
                        # the animals do not differ beyond the noise (random-intercept variance ~ 0): the mixed
                        # model is singular.  Ordinary least squares with standard errors clustered by animal
                        # keeps the repeated rows of one animal from counting as independent.
                        fit=smf.ols(formula,d).fit(cov_type="cluster",cov_kwds={"groups":pd.factorize(d["__subject"])[0]})
                        ci=fit.conf_int()
                        kind="ols"
                        estimation="OLS with animal-clustered standard errors (mixed model singular: animal variance ≈ 0)"
            # the model in the user's own column names (for the methods section)
            back={v:k for k,v in rename.items()}
            fixed=" + ".join(t.replace("C(__group)",f"{back.get('__group')}").replace("C(__time)",f"{back.get('__time')}") for t in terms)
            rand=f"(1 | {subject})"
            if vc and cage and dam and "Dam→Cage" in str(dam_status): rand+=f" + (1 | {dam}:{cage})"
            elif vc and "Dam variance" in str(dam_status): rand+=f" + (1 | {dam})"
            model_txt=f"{metric} ~ {fixed} + {rand}" if kind=="mixed" else f"{metric} ~ {fixed}  [SE clustered by {subject}]"
            try: resid_p=float(stats.shapiro(np.asarray(fit.resid)[:5000]).pvalue) if len(fit.resid)>=3 else np.nan
            except Exception: resid_p=np.nan
            for name,p in fit.pvalues.items():
                if name=="Intercept" or name in ("Group Var",) or name.endswith(" Var"): continue
                nice=name.replace("C(__group)",str(back.get("__group"))).replace("C(__time)",str(back.get("__time")))
                rows.append({"Metric":metric,"Term":nice,"Estimate":float(fit.params.get(name,np.nan)),"SE":float(fit.bse.get(name,np.nan)),"p_value":float(p),"CI95_low":float(ci.loc[name,0]) if name in ci.index else np.nan,"CI95_high":float(ci.loc[name,1]) if name in ci.index else np.nan,"N_rows":len(d),"N_subjects":d["__subject"].nunique(),"Dam_Cage":dam_status,
                             "Model":model_txt,"Estimation":estimation,"Converged":bool(getattr(fit,"converged",True)),
                             "Residual_normality_Shapiro_p":resid_p,"Status":"ok"})
        except Exception as exc:
            rows.append({"Metric":metric,"Status":f"Model failed: {exc}"})
    out=pd.DataFrame(rows)
    if "p_value" in out.columns and out["p_value"].notna().any():
        from analysis.statistics import _bh_adjust
        out["p_adjusted_BH"]=_bh_adjust(out["p_value"].to_numpy(dtype=float))
        out["Correction"]=f"Benjamini–Hochberg FDR over all {int(out['p_value'].notna().sum())} model terms"
    return out


def methods_table(df, alpha=0.05):
    """Plain-language description of how every result in the workbook was computed (for a methods section)."""
    import scipy
    try:
        import statsmodels
        sm_v=statsmodels.__version__
    except Exception:
        sm_v="not installed"
    subject=_first(df,SUBJECT_CANDIDATES); group=_first(df,GROUP_CANDIDATES); time=_first(df,TIME_CANDIDATES)
    dam=_first(df,DAM_CANDIDATES); cage=_first(df,CAGE_CANDIDATES)
    rows=[
        ("Experimental unit", f"The animal ({subject}): several videos of one animal are averaged before group tests (no pseudoreplication)." if subject else "No animal ID column: rows (videos) are the unit — add an Animal ID column."),
        ("Columns used", f"Animal: {subject or '—'} · Group: {group or '—'} · Time point: {time or '—'} · Dam: {dam or '—'} · Cage: {cage or '—'}"),
        ("Two groups", f"Welch t-test when every group looks normal (Shapiro–Wilk p ≥ {alpha:g}), else Mann–Whitney U (two-sided). Effect size: Cohen's d (pooled SD) with a bootstrap 95 % CI where shown."),
        ("Three or more groups", f"One-way ANOVA when every group looks normal (Shapiro–Wilk p ≥ {alpha:g}), else Kruskal–Wallis. Effect size: eta squared. Levene's test for equal variances is reported next to each result."),
        ("Multiple comparisons", "Benjamini–Hochberg false-discovery-rate adjustment across all metrics compared (column p_adjusted); mixed-model terms are adjusted across all terms (p_adjusted_BH). 'Significant' uses the adjusted p."),
        ("Repeated measures", "Two time points: paired t-test (Wilcoxon signed-rank when too few pairs) on the change per animal, with its 95 % CI. More time points: Friedman test on complete cases."),
        ("Mixed-effects model", "Linear mixed model (statsmodels MixedLM, maximum likelihood): metric ~ group + time + group:time, random intercept per animal; Dam (and Cage within Dam) as variance components when present and estimable. The model formula, convergence and a Shapiro–Wilk test of the residuals are printed per metric. When the animals do not differ beyond the noise (random-intercept variance ≈ 0, a singular model) the same fixed effects are fitted by OLS with standard errors clustered by animal, and the Estimation column says so."),
        ("Alpha", f"{alpha:g} (two-sided)"),
        ("Software", f"BehavioralTracker statistics · SciPy {scipy.__version__} · statsmodels {sm_v}"),
    ]
    return pd.DataFrame(rows, columns=["Item","How it was done"])

def run_advanced_statistics(design_path, output_dir, metric_cols=None):
    path=Path(design_path); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    from output.batch_summary import read_batch_summary
    df=read_batch_summary(path)
    audit=design_audit(df); subject=subject_level_summary(df,metric_cols); repeated=repeated_measures(df,metric_cols); mixed=mixed_effects(df,metric_cols)
    # Group-level independent comparison is delegated to the already design-aware engine.
    from analysis.statistics import compare_groups
    group=_first(df,GROUP_CANDIDATES)
    between=compare_groups(df,group) if group else pd.DataFrame()
    audit.to_csv(out/"statistics_design_audit.csv",index=False)
    subject.to_csv(out/"statistics_subject_level.csv",index=False)
    repeated.to_csv(out/"statistics_repeated_measures.csv",index=False)
    mixed.to_csv(out/"statistics_mixed_effects.csv",index=False)
    between.to_csv(out/"statistics_between_groups.csv",index=False)
    with pd.ExcelWriter(out/"Advanced_Statistics.xlsx",engine="openpyxl") as w:
        methods_table(df).to_excel(w,sheet_name="Methods",index=False)
        audit.to_excel(w,sheet_name="Design_Audit",index=False); subject.to_excel(w,sheet_name="Subject_Level",index=False); repeated.to_excel(w,sheet_name="Repeated_Measures",index=False); mixed.to_excel(w,sheet_name="Mixed_Effects",index=False); between.to_excel(w,sheet_name="Between_Groups",index=False)
    return {"audit":audit,"subject":subject,"repeated":repeated,"mixed":mixed,"between":between,"excel_path":str(out/"Advanced_Statistics.xlsx")}
