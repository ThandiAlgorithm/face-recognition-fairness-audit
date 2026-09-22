"""Generate a report-ready REP-001..REP-006 figure/PDF pack from repository CSV outputs.

Usage from the repository root:
    python generate_rep_report_from_repo.py

Or:
    python generate_rep_report_from_repo.py --repo "C:\\path\\to\\face-recognition-fairness-audit"

Outputs:
    results/representation/report_pack/figures/*.png
    results/representation/report_pack/REP_Audit_Report_From_Current_Results.pdf

Dependencies: pandas, numpy, matplotlib.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COL = {
    "blue": "#3E6C8A", "teal": "#4F8F8B", "gold": "#C8963E",
    "rose": "#B35B66", "navy": "#243B53", "gray": "#7B8794",
    "green": "#5B8C6A", "red": "#A64B4B", "purple": "#7467A8",
}


def read_csv(path: Path, required=True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required REP output not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        if required:
            raise
        return pd.DataFrame()


def save(fig, figures_dir: Path, name: str, pdf: PdfPages):
    fig.tight_layout()
    fig.savefig(figures_dir / name, dpi=220, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def title_page(pdf: PdfPages, snapshot: str):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(.5, .78, "REPRESENTATION AUDIT", ha="center", fontsize=26, weight="bold", color=COL["navy"])
    fig.text(.5, .735, "REP-001 through REP-006", ha="center", fontsize=17, color=COL["blue"])
    fig.text(.5, .69, "Report-ready figures and summary tables", ha="center", fontsize=13, color=COL["gray"])
    fig.text(.5, .56, f"Phase 0 snapshot: {snapshot}", ha="center", fontsize=11)
    fig.text(.5, .49, "Regenerate after the final canonical Phase 0 hand-off.", ha="center", fontsize=10, color=COL["red"])
    fig.text(.5, .20, "Generated directly from results/representation/REP-001 ... REP-006", ha="center", fontsize=9, color=COL["gray"])
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def text_page(pdf: PdfPages, title: str, paragraphs: list[str]):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(.08, .94, title, fontsize=18, weight="bold", color=COL["navy"], va="top")
    y=.88
    for para in paragraphs:
        lines=[]
        words=para.split()
        line=""
        for word in words:
            test=(line+" "+word).strip()
            if len(test)>95:
                lines.append(line); line=word
            else:
                line=test
        if line: lines.append(line)
        fig.text(.08, y, "\n".join(lines), fontsize=10.5, color="#243B53", va="top", linespacing=1.45)
        y -= .035*len(lines)+.035
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def table_page(pdf: PdfPages, title: str, df: pd.DataFrame, fontsize=8, col_widths=None):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title(title, fontsize=17, color=COL["navy"], loc="left", pad=18)
    display=df.copy()
    tbl=ax.table(cellText=display.astype(str).values, colLabels=list(display.columns), cellLoc="left", colLoc="left", loc="upper left", bbox=[0.02,0.05,0.96,0.86])
    tbl.auto_set_font_size(False); tbl.set_fontsize(fontsize)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor("#C7D7E0"); cell.set_linewidth(.5)
        if r==0:
            cell.set_facecolor("#DCE6EC"); cell.set_text_props(weight="bold", color=COL["navy"])
        elif r%2==0:
            cell.set_facecolor("#F7FAFC")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="Repository root")
    ap.add_argument("--snapshot", default="99efe8f", help="Phase 0 snapshot identifier for report cover")
    args=ap.parse_args()

    repo=Path(args.repo).resolve()
    root=repo / "results" / "representation"
    out=root / "report_pack"
    figures=out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    pdf_path=out / "REP_Audit_Report_From_Current_Results.pdf"

    # Load outputs
    rep1=root/'REP-001'; rep2=root/'REP-002'; rep3=root/'REP-003'; rep4=root/'REP-004'; rep5=root/'REP-005'; rep6=root/'REP-006'
    ds=read_csv(rep1/'rep001_dataset_summary.csv')
    gender=read_csv(rep1/'rep001_gender_composition.csv')
    race=read_csv(rep1/'rep001_race_composition.csv')
    age=read_csv(rep1/'rep001_age_band_harmonised_composition.csv')
    avail=read_csv(rep1/'rep001_demographic_availability.csv', required=False)
    ids=read_csv(rep2/'rep002_identity_support.csv')
    depth_sum=read_csv(rep3/'rep003_identity_depth_summary.csv')
    depth_dist=read_csv(rep3/'rep003_identity_depth_distribution.csv')
    int_status=read_csv(rep4/'rep004_reporting_status_summary.csv')
    int_comp=read_csv(rep4/'rep004_intersectional_composition.csv')
    overall=read_csv(rep5/'rep005_overall_base_rate.csv')
    br_race=read_csv(rep5/'rep005_base_rate_by_race.csv')
    br_age=read_csv(rep5/'rep005_base_rate_by_age.csv')
    miss=read_csv(rep6/'rep006_label_missingness.csv')
    miss_parse=read_csv(rep6/'rep006_missingness_vs_parse_failure.csv')
    affected=read_csv(rep6/'rep006_affected_analyses.csv')

    with PdfPages(pdf_path) as pdf:
        title_page(pdf,args.snapshot)
        text_page(pdf,"Executive summary",[
            "This report pack consolidates the six representation-audit issues into publication-ready figures and compact summary tables. It is generated from the machine-readable CSV outputs rather than manually typed values.",
            "Interpretation boundary: representation imbalance, thin cells, missing metadata and target prevalence are dataset findings. They are not, by themselves, proof of recognition-model bias or causal demographic effects.",
            "All results remain provisional until the final Phase 0 hand-off is merged and the full audit is rerun with the frozen governance configuration."
        ])

        # Dataset size
        d=ds.copy(); d['dataset']=d['dataset'].astype(str).str.title(); d=d.sort_values('n_images',ascending=False)
        fig,ax=plt.subplots(figsize=(8.4,5.1)); bars=ax.bar(d['dataset'],d['n_images'],color=[COL['blue'],COL['teal'],COL['gray']][:len(d)])
        ax.set_title('Dataset size'); ax.set_ylabel('Images'); ax.grid(axis='y',alpha=.18)
        for b,v in zip(bars,d['n_images']): ax.text(b.get_x()+b.get_width()/2,v*1.02,f'{int(v):,}',ha='center')
        save(fig,figures,'01_dataset_size.png',pdf)

        # Gender 100%
        g=gender.copy(); g['category']=g['category'].astype(str).str.title()
        gp=g.pivot(index='dataset',columns='category',values='percentage').fillna(0)
        fig,ax=plt.subplots(figsize=(8.4,5.2)); bottom=np.zeros(len(gp))
        for cat,c in [('Female',COL['rose']),('Male',COL['blue']),('Unknown',COL['gray'])]:
            vals=gp[cat].values if cat in gp else np.zeros(len(gp)); ax.bar(gp.index,vals,bottom=bottom,label=cat,color=c); bottom+=vals
        ax.set_ylim(0,100); ax.yaxis.set_major_formatter(PercentFormatter()); ax.set_title('Gender-label composition'); ax.set_ylabel('Share of dataset'); ax.legend(frameon=False,ncol=3); ax.grid(axis='y',alpha=.18)
        save(fig,figures,'02_gender_composition.png',pdf)

        # Age 100%
        a=age.copy(); apiv=a.pivot(index='dataset',columns='category',values='percentage').fillna(0)
        fig,ax=plt.subplots(figsize=(8.5,5.2)); bottom=np.zeros(len(apiv)); order=['0-19','20-39','40-59','60+','unknown']; cs=[COL['blue'],COL['teal'],COL['gold'],COL['purple'],COL['gray']]
        for cat,c in zip(order,cs):
            vals=apiv[cat].values if cat in apiv else np.zeros(len(apiv)); ax.bar(apiv.index,vals,bottom=bottom,label=cat,color=c); bottom+=vals
        ax.set_ylim(0,100); ax.yaxis.set_major_formatter(PercentFormatter()); ax.set_title('Harmonised age-band composition'); ax.set_ylabel('Share of dataset'); ax.legend(frameon=False,ncol=5); ax.grid(axis='y',alpha=.18)
        save(fig,figures,'03_age_composition.png',pdf)

        # Race per dataset
        for k,dsname in enumerate(sorted(race['dataset'].dropna().unique()),start=4):
            r=race[(race.dataset==dsname)&(race.category.astype(str).str.lower()!='unknown')].sort_values('n_images')
            if r.empty: continue
            fig,ax=plt.subplots(figsize=(8.6,5.3)); bars=ax.barh(r['category'],r['n_images'],color=COL['blue'] if str(dsname).lower()=='fairface' else COL['teal'])
            ax.set_title(f'{str(dsname).title()} native race-label composition'); ax.set_xlabel('Images'); ax.grid(axis='x',alpha=.18)
            save(fig,figures,f'{k:02d}_{str(dsname).lower()}_race.png',pdf)

        # LFW singleton/repeated
        l=ids[ids.dataset.astype(str).str.lower()=='lfw']
        if not l.empty:
            row=l.iloc[0]
            fig,ax=plt.subplots(figsize=(7.2,5.3)); ax.pie([row.n_singleton_identities,row.n_repeated_identities],labels=['Singleton','Repeated'],autopct=lambda p:f'{p:.1f}%',startangle=90,colors=[COL['gold'],COL['blue']],wedgeprops={'edgecolor':'white'}); ax.set_title('LFW identity depth: singleton versus repeated identities')
            save(fig,figures,'06_lfw_singletons.png',pdf)

        # depth distribution full, clipped visual x<=50 plus tail note
        ld=depth_dist[depth_dist.dataset.astype(str).str.lower()=='lfw'].copy()
        if not ld.empty:
            low=ld[ld.n_images_per_identity<=50]
            fig,ax=plt.subplots(figsize=(9,5.3)); ax.bar(low.n_images_per_identity,low.n_identities,color=COL['blue']); ax.set_yscale('log'); ax.set_xlabel('Images per identity'); ax.set_ylabel('Identities (log scale)'); ax.set_title('LFW identity-depth distribution (depths up to 50 shown)'); ax.grid(axis='y',alpha=.18,which='both'); ax.text(.99,.95,f"Full table extends to {int(ld.n_images_per_identity.max())} images/identity",transform=ax.transAxes,ha='right',va='top',color=COL['gray'])
            save(fig,figures,'07_lfw_depth_distribution.png',pdf)

        # Intersection status
        ist=int_status.copy(); ist['label']=ist.dataset.astype(str).str.title()+' - '+ist.intersection.astype(str).str.replace('_',' ',regex=False)
        fig,ax=plt.subplots(figsize=(9.2,6)); y=np.arange(len(ist)); ax.barh(y,ist.n_reportable_cells,label='Reportable',color=COL['green']); ax.barh(y,ist.n_low_confidence_cells,left=ist.n_reportable_cells,label='Low confidence',color=COL['gold']); ax.barh(y,ist.n_suppressed_cells,left=ist.n_reportable_cells+ist.n_low_confidence_cells,label='Suppressed',color=COL['red']); ax.set_yticks(y,ist.label); ax.invert_yaxis(); ax.set_xlabel('Cells'); ax.set_title('REP-004 intersection reporting status'); ax.legend(frameon=False,ncol=3); ax.grid(axis='x',alpha=.18)
        save(fig,figures,'08_intersection_status.png',pdf)

        # Flagged intersection cells, from full CSV
        flagged=int_comp[int_comp.reporting_status!='reportable'].copy()
        if not flagged.empty:
            flagged['label']=flagged['dataset'].astype(str).str.title()+' | '+flagged['cell_label'].astype(str)
            flagged=flagged.sort_values('n_images')
            fig,ax=plt.subplots(figsize=(9.4,max(4.8,.35*len(flagged)+2))); cc=[COL['red'] if s=='suppressed' else COL['gold'] for s in flagged.reporting_status]
            ax.barh(flagged.label,flagged.n_images,color=cc); ax.axvline(30,color=COL['red'],ls='--'); ax.axvline(100,color=COL['gold'],ls='--'); ax.set_title('REP-004 non-reportable and low-confidence cells'); ax.set_xlabel('Eligible images'); ax.grid(axis='x',alpha=.18)
            save(fig,figures,'09_flagged_intersection_cells.png',pdf)

        # Overall base rate
        ov=overall[overall.n>0].copy(); ov['pct']=ov.base_rate_pct_reported.fillna(ov.base_rate_pct_raw)
        fig,ax=plt.subplots(figsize=(7.6,5)); bars=ax.bar(ov.dataset.astype(str).str.title(),ov.pct,color=[COL['blue'],COL['teal']][:len(ov)]); ax.set_ylim(0,60); ax.yaxis.set_major_formatter(PercentFormatter()); ax.axhline(50,color=COL['gray'],ls='--'); ax.set_title('Overall base rate: Y=1 = female'); ax.set_ylabel('P(female)'); ax.grid(axis='y',alpha=.18)
        for b,v in zip(bars,ov.pct): ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.2f}%',ha='center')
        save(fig,figures,'10_overall_base_rate.png',pdf)

        # Race base rates per dataset
        datasets=sorted(br_race.dataset.unique())
        fig,axes=plt.subplots(1,len(datasets),figsize=(6*max(1,len(datasets)),5.3),sharex=True)
        if len(datasets)==1: axes=[axes]
        for ax,dsname in zip(axes,datasets):
            d=br_race[br_race.dataset==dsname].copy(); d['pct']=d.base_rate_pct_reported.fillna(d.base_rate_pct_raw); d=d.sort_values('pct'); ax.barh(d.subgroup,d.pct,color=COL['blue'] if str(dsname).lower()=='fairface' else COL['teal']); ax.axvline(50,color=COL['gray'],ls='--'); ax.set_title(str(dsname).title()); ax.xaxis.set_major_formatter(PercentFormatter()); ax.grid(axis='x',alpha=.18)
        fig.suptitle('Female base rate by native race category',fontsize=14)
        save(fig,figures,'11_base_rate_by_race.png',pdf)

        # Age base rates
        b=br_age.copy(); b['pct']=b.base_rate_pct_reported.fillna(b.base_rate_pct_raw)
        piv=b.pivot(index='subgroup',columns='dataset',values='pct'); order=[x for x in ['0-19','20-39','40-59','60+'] if x in piv.index]; piv=piv.reindex(order)
        fig,ax=plt.subplots(figsize=(8.5,5.2)); x=np.arange(len(piv)); width=.8/max(1,len(piv.columns));
        for i,dsname in enumerate(piv.columns): ax.bar(x-.4+width/2+i*width,piv[dsname],width=width,label=str(dsname).title(),color=[COL['blue'],COL['teal'],COL['gold']][i%3])
        ax.set_xticks(x,piv.index); ax.set_ylim(0,60); ax.yaxis.set_major_formatter(PercentFormatter()); ax.axhline(50,color=COL['gray'],ls='--'); ax.set_title('Female base rate by harmonised age band'); ax.set_ylabel('P(female | age)'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.18)
        save(fig,figures,'12_base_rate_by_age.png',pdf)

        # Missingness heatmap
        m=miss.copy(); m['pct_missing']=pd.to_numeric(m.pct_missing_or_unknown,errors='coerce'); mp=m.pivot(index='label',columns='dataset',values='pct_missing');
        fig,ax=plt.subplots(figsize=(7.6,5)); im=ax.imshow(mp.values,vmin=0,vmax=100,cmap='OrRd',aspect='auto'); ax.set_xticks(range(mp.shape[1]),[str(x).title() for x in mp.columns]); ax.set_yticks(range(mp.shape[0]),mp.index); ax.set_title('Missing / unknown demographic labels');
        for i in range(mp.shape[0]):
            for j in range(mp.shape[1]):
                v=mp.iloc[i,j]; ax.text(j,i,f'{v:.2f}%',ha='center',va='center',color='white' if v>55 else 'black')
        fig.colorbar(im,ax=ax,fraction=.035,pad=.03,label='Missing / unknown (%)')
        save(fig,figures,'13_missingness_heatmap.png',pdf)

        # Missing vs parse
        pr=miss_parse.copy();
        fig,ax=plt.subplots(figsize=(8.6,5.1)); x=np.arange(len(pr)); w=.25; ax.bar(x-w,pr.n_any_core_label_missing,width=w,label='Any core label missing',color=COL['gold']); ax.bar(x,pr.n_parse_failed,width=w,label='Parse failed',color=COL['red']); ax.bar(x+w,pr.n_missing_and_parse_failed,width=w,label='Both',color=COL['purple']); ax.set_xticks(x,[str(v).title() for v in pr.dataset]); ax.set_yscale('symlog',linthresh=1); ax.set_title('Missing labels versus metadata parse failures'); ax.set_ylabel('Images (symlog)'); ax.legend(frameon=False,ncol=3); ax.grid(axis='y',alpha=.18,which='both')
        save(fig,figures,'14_missingness_vs_parse_failure.png',pdf)

        # Support matrix from affected analyses
        af=affected.copy(); af['supported']=af.n_eligible_complete_rows>0
        sm=af.pivot(index='analysis',columns='dataset',values='supported').fillna(False)
        fig,ax=plt.subplots(figsize=(8.7,6)); im=ax.imshow(sm.astype(int).values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto'); ax.set_xticks(range(sm.shape[1]),[str(x).title() for x in sm.columns]); ax.set_yticks(range(sm.shape[0]),sm.index); ax.set_title('Demographic analysis availability after missingness/eligibility');
        for i in range(sm.shape[0]):
            for j in range(sm.shape[1]): ax.text(j,i,'Available' if sm.iloc[i,j] else 'Unavailable',ha='center',va='center',fontsize=8,color='white' if sm.iloc[i,j] else 'black')
        save(fig,figures,'15_analysis_availability.png',pdf)

        # Tables
        table_page(pdf,'Table: REP-002 identity support',ids)
        table_page(pdf,'Table: REP-003 identity-depth summary',depth_sum)
        table_page(pdf,'Table: REP-004 intersection status',int_status)
        table_page(pdf,'Table: REP-005 base rate by race',br_race[['dataset','subgroup','n','n_y1','n_y0','base_rate_pct_reported','reporting_status']])
        table_page(pdf,'Table: REP-005 base rate by age',br_age[['dataset','subgroup','n','n_y1','n_y0','base_rate_pct_reported','reporting_status']])
        table_page(pdf,'Table: REP-006 label missingness',miss)

        text_page(pdf,'Suggested report interpretation',[
            "Dataset representation should be interpreted on multiple levels rather than through aggregate image counts alone. The current outputs show substantial source imbalance, dataset-specific metadata availability, and unequal identity depth.",
            "LFW is especially informative at the identity level: most identities are represented by a single image while a small upper tail is represented by many images. This structure should be considered when later analyses resample or aggregate LFW observations.",
            "The REP-004 small-cell flags show why broad marginal balance can be misleading. UTKFace supports broad demographic comparisons but several granular intersections fall below the frozen confidence/reporting thresholds.",
            "The REP-005 base rates are prevalence quantities under the coding convention Y=1=female; they should not be described as model accuracy, fairness scores, or evidence that a class is advantaged.",
            "REP-006 preserves limitations explicitly: LFW demographic labels are unavailable in the current hand-off, FairFace lacks exact numeric age, and four UTKFace demographic-missing rows coincide exactly with parse failures."
        ])

    print(f"Created: {pdf_path}")
    print(f"Figures: {figures}")

if __name__ == '__main__':
    main()
