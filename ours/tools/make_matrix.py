import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
runs=["Deterministic","T = 0.05","T = 0.2 (paper)","T = 0.5","T = 1.0"]
d={"MVE":[6.95514e-4,6.95060e-4,7.05643e-4,8.53708e-4,1.283668e-3],
"LVE":[6.36365e-5,6.34579e-5,6.41746e-5,8.05927e-5,1.439299e-4],
"MEE":[6.36365e-5,6.21872e-5,5.91830e-5,6.55589e-5,1.187833e-4],
"CE":[6.36365e-5,5.96757e-5,5.68775e-5,6.40567e-5,1.034013e-4],
"FDD":[4.64974e-7,4.67903e-7,5.36755e-7,8.63286e-7,1.142466e-6],
"Diversity":[0.0,1.52558e-4,3.11565e-4,5.80017e-4,7.89263e-4]}
scale={"MVE":1e-3,"LVE":1e-4,"MEE":1e-4,"CE":1e-4,"FDD":1e-5,"Diversity":1e-3}
paper={"MVE":0.7243,"LVE":0.6040,"MEE":0.5549,"CE":0.5227,"FDD":0.0415,"Diversity":0.3274}
cols=[("MVE","Error","lower"),("LVE","Error","lower"),("MEE","Fidelity (emotion)","lower"),("CE","Fidelity (closest sample)","lower"),("FDD","Fidelity (dynamics)","lower"),("Diversity","Variance","higher")]
with open("matrix.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["Run"]+[f"{k} (x{scale[k]:.0e})" for k,_,_ in cols])
    for i,r in enumerate(runs): w.writerow([r]+[f"{d[k][i]/scale[k]:.4f}" for k,_,_ in cols])
    w.writerow(["Paper Table 1 (T=0.2, 32 spk)"]+[f"{paper[k]:.4f}" for k,_,_ in cols])
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
fig,ax=plt.subplots(figsize=(13,5.2))
rows=runs+["Paper (T=0.2)"]
M=np.array([[d[k][i]/scale[k] for k,_,_ in cols] for i in range(5)]+[[paper[k] for k,_,_ in cols]])
# colour: per column rank, good=teal, bad=coral (exclude paper row from scaling)
from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list("g",["#2bb3a3","#f4f1ea","#e0685a"])
C=np.zeros_like(M)
for j,(k,_,better) in enumerate(cols):
    v=M[:5,j]; lo,hi=v.min(),v.max(); n=(M[:,j]-lo)/(hi-lo+1e-12)
    C[:,j]=n if better=="lower" else 1-n
C=np.clip(C,0,1); C[5,:]=np.nan
ax.imshow(np.ma.masked_invalid(C),cmap=cm,aspect="auto",vmin=0,vmax=1)
for i in range(6):
    for j in range(6):
        if i==5: ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,color="#e8e8ef"))
        ax.text(j,i,f"{M[i,j]:.4f}",ha="center",va="center",fontsize=12,fontweight="bold" if i==2 else None)
ax.set_xticks(range(6)); ax.set_xticklabels([f"{k} ({'↓' if b=='lower' else '↑'})\n{g}\n×{scale[k]:.0e}" for k,g,b in cols],fontsize=10)
ax.set_yticks(range(6)); ax.set_yticklabels(rows,fontsize=11)
ax.xaxis.tick_top(); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.axhline(4.5,color="#333",lw=1.5)
ax.set_title("Variance–Error–Fidelity matrix: ProbTalk3D on 3DMEAD ground truth (8 speakers, 478 seq, 57,715 frames)\nGreen = better within our 5 runs, red = worse. Grey row = paper Table 1.",fontsize=12,pad=70)
plt.tight_layout(); plt.savefig("variance_error_fidelity_matrix.png",dpi=180,bbox_inches="tight")
# tradeoff plot
fig,ax=plt.subplots(figsize=(7.5,5))
x=[v/1e-3 for v in d["Diversity"]]; y=[v/1e-3 for v in d["MVE"]]
ax.plot(x,y,"-o",color="#7c6cff",lw=2,ms=9)
for xi,yi,r in zip(x,y,runs): ax.annotate(r,(xi,yi),xytext=((8,-14) if ("0.05" in r or "0.2" in r) else (8,6)),textcoords="offset points",fontsize=10)
ax.plot(paper["Diversity"],paper["MVE"],"*",color="#e45cf0",ms=16,label="Paper Table 1 (T=0.2)")
ax.set_xlabel("Diversity ×10⁻³  (variance, higher = more varied takes)"); ax.set_ylabel("MVE ×10⁻³  (error, lower = closer to ground truth)")
ax.set_title("Temperature trades error for diversity"); ax.grid(alpha=.3); ax.legend(loc="upper left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("tradeoff_error_vs_diversity.png",dpi=180)
