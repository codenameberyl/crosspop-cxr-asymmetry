# Results Summary - crosspop-cxr-asymmetry

## SQ1 Discrimination
- K2N: AUROC 0.624 [0.577, 0.670]
- N2K: AUROC 0.662 [0.654, 0.669]
- Direction contrast (AUROC): 0.025 [-0.055, 0.11] -> not reliable

## SQ2 Calibration
- K2N (raw): Standard ECE 0.249 [0.215, 0.291] | Adaptive ECE 0.238 [0.205, 0.283]
- K2N (calibrated): Standard ECE 0.304 [0.271, 0.349] | Adaptive ECE 0.290 [0.258, 0.336]
- N2K (raw): Standard ECE 0.143 [0.140, 0.147] | Adaptive ECE 0.143 [0.140, 0.147]
- N2K (calibrated): Standard ECE 0.139 [0.136, 0.142] | Adaptive ECE 0.139 [0.136, 0.142]
- Direction Contrast (ECE (Equal-Width)): -0.128 [-0.193, -0.067] -> reliable
- Direction Contrast (ECE (Adaptive)): -0.112 [-0.18, -0.047] -> reliable
- Size x Direction Interaction (ECE (Equal-Width)): -0.032 [-0.093, 0.034] -> not reliable
- Size x Direction Interaction (ECE (Adaptive)): -0.018 [-0.082, 0.047] -> not reliable

## SQ3 Size Interaction
- Size x Direction Interaction (AUROC): -0.041 [-0.102, 0.02] -> none
- Size x Direction Interaction (ECE (Equal-Width)): -0.032 [-0.093, 0.034] -> none
- Size x Direction Interaction (ECE (Adaptive)): -0.018 [-0.082, 0.047] -> none

## Exploratory OOD (Musa TB/COVID)
- Mean confidence in-distribution:  0.883
- Mean confidence OOD:             0.895
- OOD-detection AUROC:             0.484 (0.5 = no separation)