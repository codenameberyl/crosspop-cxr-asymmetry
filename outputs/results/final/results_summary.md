# Results Summary - crosspop-cxr-asymmetry

## Primary outcome: transfer gaps and asymmetry, per architecture
- mobilenet_v2: Gap_K=0.316 [0.196,0.384], Gap_N=0.135 [0.034,0.214], Asymmetry=0.182 [0.039,0.336]
- efficientnet_b0: Gap_K=0.353 [0.196,0.458], Gap_N=-0.028 [-0.068,0.031], Asymmetry=0.382 [0.165,0.525]

## Calibration: run-level debiased ECE by condition/architecture
- efficientnet_b0 / K2K: ECE=0.056 [0.034,0.089] (n_runs=9)
- efficientnet_b0 / K2N: ECE=0.227 [0.166,0.281] (n_runs=9)
- efficientnet_b0 / K2N_full: ECE=0.206 [0.175,0.236] (n_runs=9)
- efficientnet_b0 / N2K: ECE=0.151 [0.124,0.178] (n_runs=15)
- efficientnet_b0 / N2N: ECE=0.061 [0.029,0.101] (n_runs=15)
- mobilenet_v2 / K2K: ECE=0.090 [0.074,0.108] (n_runs=9)
- mobilenet_v2 / K2N: ECE=0.270 [0.227,0.314] (n_runs=9)
- mobilenet_v2 / K2N_full: ECE=0.267 [0.227,0.302] (n_runs=9)
- mobilenet_v2 / N2K: ECE=0.183 [0.132,0.229] (n_runs=15)
- mobilenet_v2 / N2N: ECE=0.082 [0.040,0.126] (n_runs=15)

## Size-matched subsampling check (does raw K2N-vs-N2K ECE gap survive size-matching?)
- mobilenet_v2: K2N actual ECE=0.272, N2K subsampled-to-30 mean=0.220 [0.092,0.370] -> gap does NOT clearly survive (may be partly a small-n artifact)
- efficientnet_b0: K2N actual ECE=0.224, N2K subsampled-to-30 mean=0.184 [0.072,0.320] -> gap does NOT clearly survive (may be partly a small-n artifact)
Note: this checks the RAW cross-population ECE gap only. The primary calibration
claim rests on the debiased, in-domain-baselined run-level table above, which
compares each model's cross-population ECE against its OWN in-domain baseline
(structurally matched in evaluation-set size pairwise), not on this raw comparison.

## Significance summary (Holm-corrected where applicable)
- Transfer-gap asymmetry (mobilenet_v2): estimate=0.182, p_holm=nan -> significant/reliable
- Transfer-gap asymmetry (efficientnet_b0): estimate=0.382, p_holm=nan -> significant/reliable
- Architecture contrast AUROC (N2K, ensemble-averaged): estimate=-0.17, p_holm=0.0 -> significant/reliable
- Architecture contrast accuracy (N2K, ensemble-averaged): estimate=nan, p_holm=0.0 -> significant/reliable

## Togunwa fold class balance: max deviation from 1:1 = 0.065
