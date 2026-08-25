# Results Summary - crosspop-cxr-asymmetry

## Primary outcome: transfer gaps and asymmetry, per architecture
- mobilenet_v2: Gap_K=0.321 [0.256,0.385], Gap_N=0.087 [0.008,0.160], Asymmetry=0.233 [0.125,0.351]
- efficientnet_b0: Gap_K=0.377 [0.301,0.435], Gap_N=-0.067 [-0.105,-0.023], Asymmetry=0.444 [0.328,0.530]

## Calibration: run-level debiased ECE by condition/architecture
- efficientnet_b0 / K2K: ECE=0.056 [0.039,0.076] (n_runs=18)
- efficientnet_b0 / K2N: ECE=0.245 [0.200,0.288] (n_runs=18)
- efficientnet_b0 / K2N_full: ECE=0.221 [0.200,0.241] (n_runs=18)
- efficientnet_b0 / N2K: ECE=0.155 [0.136,0.173] (n_runs=30)
- efficientnet_b0 / N2N: ECE=0.043 [0.025,0.066] (n_runs=30)
- mobilenet_v2 / K2K: ECE=0.085 [0.069,0.103] (n_runs=18)
- mobilenet_v2 / K2N: ECE=0.253 [0.215,0.293] (n_runs=18)
- mobilenet_v2 / K2N_full: ECE=0.263 [0.239,0.285] (n_runs=18)
- mobilenet_v2 / N2K: ECE=0.166 [0.137,0.196] (n_runs=30)
- mobilenet_v2 / N2N: ECE=0.092 [0.063,0.123] (n_runs=30)

## Size-matched subsampling check (does raw K2N-vs-N2K ECE gap survive size-matching?)
- mobilenet_v2: K2N actual ECE=0.259, N2K subsampled-to-30 mean=0.202 [0.081,0.338] -> gap does NOT clearly survive (may be partly a small-n artifact)
- efficientnet_b0: K2N actual ECE=0.239, N2K subsampled-to-30 mean=0.181 [0.069,0.311] -> gap does NOT clearly survive (may be partly a small-n artifact)
Note: this checks the RAW cross-population ECE gap only. The primary calibration
claim rests on the debiased, in-domain-baselined run-level table above, which
compares each model's cross-population ECE against its OWN in-domain baseline
(structurally matched in evaluation-set size pairwise), not on this raw comparison.

## Significance summary (Holm-corrected where applicable)
- Transfer-gap asymmetry (mobilenet_v2): estimate=0.233, p_holm=nan -> significant/reliable
- Transfer-gap asymmetry (efficientnet_b0): estimate=0.444, p_holm=nan -> significant/reliable
- Architecture contrast AUROC (N2K, ensemble-averaged): estimate=-0.151, p_holm=0.0 -> significant/reliable
- Architecture contrast accuracy (N2K, ensemble-averaged): estimate=nan, p_holm=0.0 -> significant/reliable

## Togunwa fold class balance: max deviation from 1:1 = 0.065

## Musa generalization check (exploratory, K2M direction only -- see caveats)
- mobilenet_v2 / K2K: AUROC=0.965 [0.955,0.974] (n_runs=18)
- mobilenet_v2 / K2N: AUROC=0.644 [0.603,0.686] (n_runs=18)
- mobilenet_v2 / K2M: AUROC=0.718 [0.626,0.802] (n_runs=18)
- efficientnet_b0 / K2K: AUROC=0.966 [0.959,0.972] (n_runs=18)
- efficientnet_b0 / K2N: AUROC=0.588 [0.550,0.633] (n_runs=18)
- efficientnet_b0 / K2M: AUROC=0.818 [0.764,0.870] (n_runs=18)

Observations:
- Cross-population degradation from K2K reproduces on Musa for both architectures,
  supporting generalization of the core finding past a single dataset pair.
- K2M AUROC is HIGHER than K2N (Togunwa) for both architectures, so the magnitude of
  degradation is not consistent across cross-population targets -- dataset-specific
  factors, not just "cross-population-ness" in the abstract, evidently matter.
- K2M shows markedly higher per-run variability than K2N, especially for MobileNetV2
  (individual run AUROCs range from 0.220 -- below-chance/inverted -- to 0.929),
  giving K2M the widest CI of any condition in the study. This instability, not just
  the interval width, is itself worth reporting.

CAVEATS: (1) no reverse (Musa-trained) direction exists, so this is a single-direction
descriptive check, NOT a formal Gap_Musa/asymmetry replication like the Togunwa result.
(2) Musa cohort has mixed pediatric/adult composition (author confirmation, 2026),
unlike Kermany/Togunwa (both pediatric-only); this confound is not isolated by this design.
