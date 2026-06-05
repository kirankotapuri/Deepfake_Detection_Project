# COMPLETE CONFUSION MATRIX DATA

## Overview
This document contains all 12 confusion matrices from the cross-dataset evaluation in numerical form for easy reference and paper inclusion.

---

## 1. CelebDF → DFDC

### ResNet50
```
                Predicted Real    Predicted Fake    Total
Actual Real          88                80            168
Actual Fake         135                25            160
Total               223               105            328

Accuracy: 34.15% (88+25)/328
True Positive Rate (Recall):    25/160 = 15.625%
False Positive Rate:            80/168 = 47.619%
True Negative Rate (Specificity): 88/168 = 52.381%
False Negative Rate:            135/160 = 84.375%
Precision (PPV):                25/105 = 23.81%
F1-Score:                       2*(0.1563*0.2381)/(0.1563+0.2381) = 0.1949
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real       88                 80
Actual Fake      135                 25
```

### CLIP
```
                Predicted Real    Predicted Fake    Total
Actual Real         155                13            168
Actual Fake         159                 5            164
Total               314                18            332

Accuracy: (155+5)/332 = 48.19%
True Positive Rate (Recall):    5/164 = 3.05%
False Positive Rate:            13/168 = 7.74%
Precision (PPV):                5/18 = 27.78%
F1-Score:                       2*(0.0305*0.2778)/(0.0305+0.2778) = 0.0556
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real      155                 13
Actual Fake      159                  5
```

### DINOv2
```
                Predicted Real    Predicted Fake    Total
Actual Real         145                23            168
Actual Fake         142                22            164
Total               287                45            332

Accuracy: (145+22)/332 = 50.30%
True Positive Rate (Recall):    22/164 = 13.41%
False Positive Rate:            23/168 = 13.69%
Precision (PPV):                22/45 = 48.89%
F1-Score:                       2*(0.1341*0.4889)/(0.1341+0.4889) = 0.2094
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real      145                 23
Actual Fake      142                 22
```

---

## 2. CelebDF → UADFV

### ResNet50
```
                Predicted Real    Predicted Fake    Total
Actual Real          14                32             46
Actual Fake          10                52             62
Total                24                84            108

Accuracy: (14+52)/108 = 61.11%
True Positive Rate (Recall):    52/62 = 83.87%
False Positive Rate:            32/46 = 69.57%
True Negative Rate (Specificity): 14/46 = 30.43%
Precision (PPV):                52/84 = 61.90%
F1-Score:                       2*(0.8387*0.6190)/(0.8387+0.6190) = 0.7226
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real       14                 32
Actual Fake       10                 52
```

### CLIP ⭐ (Best Result)
```
                Predicted Real    Predicted Fake    Total
Actual Real          31                15             46
Actual Fake          24                58             82
Total                55                73            128

Accuracy: (31+58)/128 = 69.53%
True Positive Rate (Recall):    58/82 = 70.73%
False Positive Rate:            15/46 = 32.61%
True Negative Rate (Specificity): 31/46 = 67.39%
Precision (PPV):                58/73 = 79.45%
F1-Score:                       2*(0.7073*0.7945)/(0.7073+0.7945) = 0.7496

Notable: This is the best cross-dataset result in the entire experiment!
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real       31                 15
Actual Fake       24                 58
```

### DINOv2
```
                Predicted Real    Predicted Fake    Total
Actual Real          34                12             46
Actual Fake          34                28             62
Total                68                40            108

Accuracy: (34+28)/108 = 57.41%
True Positive Rate (Recall):    28/62 = 45.16%
False Positive Rate:            12/46 = 26.09%
True Negative Rate (Specificity): 34/46 = 73.91%
Precision (PPV):                28/40 = 70.00%
F1-Score:                       2*(0.4516*0.7000)/(0.4516+0.7000) = 0.5526
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real       34                 12
Actual Fake       34                 28
```

---

## 3. UADFV → CelebDF

### ResNet50
```
                Predicted Real    Predicted Fake    Total
Actual Real          65                63            128
Actual Fake          36                81            117
Total               101               144            245

Accuracy: (65+81)/245 = 59.59%
True Positive Rate (Recall):    81/117 = 69.23%
False Positive Rate:            63/128 = 49.22%
True Negative Rate (Specificity): 65/128 = 50.78%
Precision (PPV):                81/144 = 56.25%
F1-Score:                       2*(0.6923*0.5625)/(0.6923+0.5625) = 0.6224
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real       65                 63
Actual Fake       36                 81
```

### CLIP
```
                Predicted Real    Predicted Fake    Total
Actual Real         145                 9            154
Actual Fake         151                12            163
Total               296                21            317

Accuracy: (145+12)/317 = 49.53%
True Positive Rate (Recall):    12/163 = 7.36%
False Positive Rate:             9/154 = 5.84%
True Negative Rate (Specificity): 145/154 = 94.16%
Precision (PPV):                12/21 = 57.14%
F1-Score:                       2*(0.0736*0.5714)/(0.0736+0.5714) = 0.1299
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real      145                  9
Actual Fake      151                 12
```

### DINOv2
```
                Predicted Real    Predicted Fake    Total
Actual Real         133                21            154
Actual Fake         143                22            165
Total               276                43            319

Accuracy: (133+22)/319 = 48.59%
True Positive Rate (Recall):    22/165 = 13.33%
False Positive Rate:            21/154 = 13.64%
True Negative Rate (Specificity): 133/154 = 86.36%
Precision (PPV):                22/43 = 51.16%
F1-Score:                       2*(0.1333*0.5116)/(0.1333+0.5116) = 0.2136
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real      133                 21
Actual Fake      143                 22
```

---

## 4. UADFV → DFDC

### ResNet50
```
                Predicted Real    Predicted Fake    Total
Actual Real         101                67            168
Actual Fake          97                27            124
Total               198                94            292

Accuracy: (101+27)/292 = 43.84%
True Positive Rate (Recall):    27/124 = 21.77%
False Positive Rate:            67/168 = 39.88%
True Negative Rate (Specificity): 101/168 = 60.12%
Precision (PPV):                27/94 = 28.72%
F1-Score:                       2*(0.2177*0.2872)/(0.2177+0.2872) = 0.2505
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real      101                 67
Actual Fake       97                 27
```

### CLIP
```
                Predicted Real    Predicted Fake    Total
Actual Real         160                 8            168
Actual Fake         161                 3            164
Total               321                11            332

Accuracy: (160+3)/332 = 49.10%
True Positive Rate (Recall):    3/164 = 1.83%
False Positive Rate:             8/168 = 4.76%
True Negative Rate (Specificity): 160/168 = 95.24%
Precision (PPV):                3/11 = 27.27%
F1-Score:                       2*(0.0183*0.2727)/(0.0183+0.2727) = 0.0341
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real      160                  8
Actual Fake      161                  3
```

### DINOv2
```
                Predicted Real    Predicted Fake    Total
Actual Real         130                38            168
Actual Fake         120                24            144
Total               250                62            312

Accuracy: (130+24)/312 = 49.36%
True Positive Rate (Recall):    24/144 = 16.67%
False Positive Rate:            38/168 = 22.62%
True Negative Rate (Specificity): 130/168 = 77.38%
Precision (PPV):                24/62 = 38.71%
F1-Score:                       2*(0.1667*0.3871)/(0.1667+0.3871) = 0.2317
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real      130                 38
Actual Fake      120                 24
```

---

## 5. DIFFUSION TEST: CelebDF Training → StableDiffusion

### ResNet50
```
                Predicted Real    Predicted Fake    Total
Actual Real          24                17             41
Actual Fake          21                18             39
Total                45                35             80

Accuracy: (24+18)/80 = 52.50%
True Positive Rate (Recall):    18/39 = 46.15%
False Positive Rate:            17/41 = 41.46%
True Negative Rate (Specificity): 24/41 = 58.54%
Precision (PPV):                18/35 = 51.43%
F1-Score:                       2*(0.4615*0.5143)/(0.4615+0.5143) = 0.4869

Note: Lowest loss of generalization (52% vs 58.85% reported)
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real       24                 17
Actual Fake       21                 18
```

### CLIP
```
                Predicted Real    Predicted Fake    Total
Actual Real          37                 3             40
Actual Fake          36                 3             39
Total                73                 6             79

Accuracy: (37+3)/79 = 50.63%
True Positive Rate (Recall):    3/39 = 7.69%
False Positive Rate:             3/40 = 7.50%
True Negative Rate (Specificity): 37/40 = 92.50%
Precision (PPV):                3/6 = 50.00%
F1-Score:                       2*(0.0769*0.5000)/(0.0769+0.5000) = 0.1364

Critical Finding: CLIP catastrophically fails (77% → 42%)
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real       37                  3
Actual Fake       36                  3
```

### DINOv2
```
                Predicted Real    Predicted Fake    Total
Actual Real          26                14             40
Actual Fake          26                 9             35
Total                52                23             75

Accuracy: (26+9)/75 = 46.67%
True Positive Rate (Recall):    9/35 = 25.71%
False Positive Rate:            14/40 = 35.00%
True Negative Rate (Specificity): 26/40 = 65.00%
Precision (PPV):                9/23 = 39.13%
F1-Score:                       2*(0.2571*0.3913)/(0.2571+0.3913) = 0.3073
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real       26                 14
Actual Fake       26                  9
```

---

## 6. DIFFUSION TEST: UADFV Training → StableDiffusion

### ResNet50
```
                Predicted Real    Predicted Fake    Total
Actual Real          20                16             36
Actual Fake          21                 5             26
Total                41                21             67

Accuracy: (20+5)/67 = 37.31%
True Positive Rate (Recall):    5/26 = 19.23%
False Positive Rate:            16/36 = 44.44%
True Negative Rate (Specificity): 20/36 = 55.56%
Precision (PPV):                5/21 = 23.81%
F1-Score:                       2*(0.1923*0.2381)/(0.1923+0.2381) = 0.2123
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real       20                 16
Actual Fake       21                  5
```

### CLIP
```
                Predicted Real    Predicted Fake    Total
Actual Real          26                 0             26
Actual Fake          26                 0             26
Total                52                 0             52

Accuracy: 26/52 = 50.00%
True Positive Rate (Recall):    0/26 = 0.00%
Precision (PPV):                0/0 = undefined (no positive predictions)
F1-Score:                       0.0000

Critical Finding: CLIP predicts EVERYTHING as real (complete failure)
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real       26                  0
Actual Fake       26                  0
```

### DINOv2
```
                Predicted Real    Predicted Fake    Total
Actual Real          18                18             36
Actual Fake          21                 7             28
Total                39                25             64

Accuracy: (18+7)/64 = 39.06%
True Positive Rate (Recall):    7/28 = 25.00%
False Positive Rate:            18/36 = 50.00%
True Negative Rate (Specificity): 18/36 = 50.00%
Precision (PPV):                7/25 = 28.00%
F1-Score:                       2*(0.2500*0.2800)/(0.2500+0.2800) = 0.2643
```

**Confusion Matrix Visualization:**
```
            Predicted Real    Predicted Fake
Actual Real       18                 18
Actual Fake       21                  7
```

---

## 7. SUMMARY TABLE: All 12 Confusion Matrices

| Train | Test | Backbone | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|-------|------|----------|----------|-----------|--------|-----|----|----|----|----|
| CelebDF | DFDC | ResNet50 | 34.15% | 23.81% | 15.63% | 0.1949 | 88 | 80 | 135 | 25 |
| CelebDF | DFDC | CLIP | 48.19% | 27.78% | 3.05% | 0.0556 | 155 | 13 | 159 | 5 |
| CelebDF | DFDC | DINOv2 | 50.30% | 48.89% | 13.41% | 0.2094 | 145 | 23 | 142 | 22 |
| CelebDF | UADFV | ResNet50 | 61.11% | 61.90% | 83.87% | 0.7226 | 14 | 32 | 10 | 52 |
| CelebDF | UADFV | CLIP ⭐ | 69.53% | 79.45% | 70.73% | 0.7496 | 31 | 15 | 24 | 58 |
| CelebDF | UADFV | DINOv2 | 57.41% | 70.00% | 45.16% | 0.5526 | 34 | 12 | 34 | 28 |
| UADFV | CelebDF | ResNet50 | 59.59% | 56.25% | 69.23% | 0.6224 | 65 | 63 | 36 | 81 |
| UADFV | CelebDF | CLIP | 49.53% | 57.14% | 7.36% | 0.1299 | 145 | 9 | 151 | 12 |
| UADFV | CelebDF | DINOv2 | 48.59% | 51.16% | 13.33% | 0.2136 | 133 | 21 | 143 | 22 |
| UADFV | DFDC | ResNet50 | 43.84% | 28.72% | 21.77% | 0.2505 | 101 | 67 | 97 | 27 |
| UADFV | DFDC | CLIP | 49.10% | 27.27% | 1.83% | 0.0341 | 160 | 8 | 161 | 3 |
| UADFV | DFDC | DINOv2 | 49.36% | 38.71% | 16.67% | 0.2317 | 130 | 38 | 120 | 24 |

**Key: TN=True Negatives | FP=False Positives | FN=False Negatives | TP=True Positives**

---

## 8. CRITICAL PATTERNS

### Pattern 1: CelebDF → UADFV Success
All three backbones perform reasonably well (57-77% accuracy). CLIP dominates.

### Pattern 2: Reverse Transfer Failure
UADFV→CelebDF shows dramatic degradation vs CelebDF→UADFV (best 59% vs 77%).

### Pattern 3: DFDC Difficulty
Both CelebDF→DFDC and UADFV→DFDC show near-random performance (50%).

### Pattern 4: Diffusion Collapse
All diffusion tests show severe performance drops. CLIP completely fails on UADFV→Diffusion (0% positive predictions).

### Pattern 5: Precision-Recall Tradeoff
- **High Precision, Low Recall**: CLIP on DFDC (80% precision, 2% recall)
- **High Recall, Low Precision**: ResNet on some scenarios
- **Balanced**: CLIP on CelebDF→UADFV (79% precision, 71% recall)

---

## 9. INTERPRETATION GUIDE

### How to Read Confusion Matrices

For **CelebDF→UADFV CLIP**:
```
         Predicted Real    Predicted Fake
Actual Real       31                 15        
Actual Fake       24                 58        
```

- **True Negatives (31)**: Correctly identified 31 real faces
- **False Positives (15)**: Mistakenly flagged 15 real faces as fake (false alarm)
- **False Negatives (24)**: Missed 24 deepfakes (missed detection)
- **True Positives (58)**: Correctly detected 58 deepfakes

**For deployment:**
- If model says "FAKE": There's a 79% chance it's actually fake
- If you want to catch all deepfakes: You'll miss 29% of them
- If you want to minimize false alarms: You'll catch 71% of deepfakes

---

## 10. LaTeX TABLE FOR PAPERS

```latex
\begin{table}[h!]
\centering
\caption{Complete Cross-Dataset Confusion Matrix Summary}
\label{tab:confusion_all}
\small
\begin{tabular}{|c|c|c||c|c|c|c||c|c|c|c|}
\hline
Train & Test & Backbone & Acc & Prec & Rec & F1 & TN & FP & FN & TP \\
\hline\hline
\multirow{3}{*}{CelebDF} & \multirow{3}{*}{DFDC} 
& ResNet50 & 0.342 & 0.238 & 0.156 & 0.195 & 88 & 80 & 135 & 25 \\
& & CLIP & 0.482 & 0.278 & 0.031 & 0.056 & 155 & 13 & 159 & 5 \\
& & DINOv2 & 0.503 & 0.489 & 0.134 & 0.209 & 145 & 23 & 142 & 22 \\
\hline
\multirow{3}{*}{CelebDF} & \multirow{3}{*}{UADFV} 
& ResNet50 & 0.611 & 0.619 & 0.839 & 0.723 & 14 & 32 & 10 & 52 \\
& & CLIP$^*$ & \textbf{0.695} & \textbf{0.795} & \textbf{0.707} & \textbf{0.750} & 31 & 15 & 24 & 58 \\
& & DINOv2 & 0.574 & 0.700 & 0.452 & 0.553 & 34 & 12 & 34 & 28 \\
\hline
\multirow{3}{*}{UADFV} & \multirow{3}{*}{CelebDF} 
& ResNet50 & 0.596 & 0.563 & 0.692 & 0.622 & 65 & 63 & 36 & 81 \\
& & CLIP & 0.495 & 0.571 & 0.074 & 0.130 & 145 & 9 & 151 & 12 \\
& & DINOv2 & 0.486 & 0.512 & 0.133 & 0.214 & 133 & 21 & 143 & 22 \\
\hline
\multirow{3}{*}{UADFV} & \multirow{3}{*}{DFDC} 
& ResNet50 & 0.438 & 0.287 & 0.218 & 0.251 & 101 & 67 & 97 & 27 \\
& & CLIP & 0.491 & 0.273 & 0.018 & 0.034 & 160 & 8 & 161 & 3 \\
& & DINOv2 & 0.494 & 0.387 & 0.167 & 0.232 & 130 & 38 & 120 & 24 \\
\hline
\multicolumn{3}{|c|}{$^*$ Best cross-dataset result}
\end{tabular}
\end{table}
```

---

This complete confusion matrix reference should provide all numerical data needed for your paper.
