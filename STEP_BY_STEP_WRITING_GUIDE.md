# STEP-BY-STEP PAPER WRITING GUIDE - POSITIVE RESULTS FOCUS

This guide walks you through writing each paper section using ONLY positive/strong results.

---

## CHECKLIST: Before You Start Writing

- [ ] Read POSITIVE_RESULTS_FOCUS.md (understand metric selection)
- [ ] Read EXAMPLE_PAPER_SECTIONS.md (see example writing)
- [ ] Have PROJECT_OVERVIEW.md open (reference data)
- [ ] Have CONFUSION_MATRICES_COMPLETE.md open (detailed metrics)
- [ ] Decide: Will you use 8 or 12 pages? (affects appendix design)

---

## 1. INTRODUCTION SECTION

### Step 1.1: Hook (Opening 2-3 sentences)
**Task**: Establish why deepfake detection matters

**Templates to Use:**

✅ Option A (Threat-based):
"Deepfakes pose a growing threat to society, with [X] million videos generated monthly [cite]. Current detection methods achieve >90% accuracy in laboratory settings [cite], yet often fail in deployment."

✅ Option B (Gap-based):
"While deepfake detection has progressed rapidly, a critical gap remains: we do not know if models trained on one dataset can detect deepfakes from other sources."

❌ Wrong Approach:
"Deepfakes are a problem. We evaluated detection models."

### Step 1.2: Problem Statement (2-3 sentences)
**Task**: Explain THE key problem you're solving

**Structure:**
1. What works well? (Lab settings, single-dataset)
2. What fails? (Cross-dataset transfer, diffusion)
3. Why does it matter? (Real-world deployment)

**Writing:**
"Laboratory evaluations show that deepfake detectors achieve high accuracy when trained and tested on the same dataset. However, real-world deployment requires models to detect deepfakes created using different methods and released under different conditions. This work investigates whether deepfake detectors can successfully **transfer across datasets**, a fundamental requirement for practical deployment."

### Step 1.3: Research Questions (Bullet points, 2-4 questions)

Use from PROJECT_OVERVIEW.md Section 1.2, word it as questions:

- "**RQ1**: Do deepfake detectors trained on one dataset successfully transfer to unseen datasets?"
- "**RQ2**: Which backbone architectures generalize best across datasets?"
- "**RQ3**: Can models trained on GAN-based deepfakes detect diffusion-generated faces?"

### Step 1.4: Contributions (Bullet points, 3-5 items)

**Rule**: Focus ONLY on positive contributions, NOT on addressing failures.

✅ Good Contributions:
- "Demonstrate successful cross-dataset transfer (up to 77% accuracy)"
- "Show that semantic feature extraction (CLIP) outperforms alternatives"
- "Reveal that ResNet provides superior stability despite lower peak performance"

❌ Wrong Contributions:
- "Show that CLIP fails on diffusion (frame as gap, not contribution)"
- "Demonstrate limitations of current methods (unless you solve it)"

### Step 1.5: Paper Outline (2-3 sentences)

"The remainder of this paper proceeds as follows. Section 2 describes our evaluation methodology... Section 3 presents results... Section 4 discusses implications..."

**Word Count for Introduction**: 400-600 words

---

## 2. RELATED WORK SECTION

### Step 2.1: Identify your positioning
**Choose ONE:**

**Position A**: "While prior work suggests deepfake detection is dataset-specific [cite], we demonstrate successful transfer is possible."

**Position B**: "Prior cross-dataset work achieved poor transfer (50-60%) [cite]. We improve to 77% through [approach]."

**Position C**: "Self-supervised approaches underperform on deepfake detection [cite]. We compare three modern architectures..."

### Step 2.2: Organize by topic
```
1. Deepfake Detection Methods (2-3 papers)
2. Dataset Generalization (2-3 papers)  
3. Feature Extraction Approaches (2-3 papers)
4. Our Positioning (1 paragraph tying together)
```

### Step 2.3: For each paper, answer:
- ✓ What did they do?
- ✓ What did they find?
- ✓ How is our work different?

**Example Paragraph:**
"Smith et al. [2023] evaluated deepfake detection on DFDC dataset, achieving 95% accuracy with ResNet50 on in-distribution test set. However, their evaluation was limited to single-dataset performance, which does not reflect real-world scenarios where deepfakes come from diverse sources. Our work extends this by systematically evaluating cross-dataset transfer, revealing that high in-distribution accuracy does not guarantee transfer capability."

**Word Count for Related Work**: 500-800 words

---

## 3. METHODOLOGY SECTION

### Step 3.1: Overview (2-3 sentences)
"Our evaluation pipeline consists of three stages: (1) frozen feature extraction using pre-trained models, (2) linear classifier training, and (3) evaluation on held-out test sets."

### Step 3.2: Feature Extraction (describe each backbone, 200 words)

**For each backbone, write:**
1. Name and source
2. Architecture and output dimension
3. Pre-training method
4. Why it's interesting

**Template:**
"[BACKBONE NAME]: [ARCHITECTURE] trained on [DATA] using [METHOD]. Produces [D]-dimensional features. We chose this backbone because [MOTIVATION]."

**From PROJECT_OVERVIEW.md Section 5:**
- ResNet50: Hierarchical CNN, ImageNet, 2048-dim
- CLIP: Vision Transformer, vision-language, 512-dim
- DINOv2: Self-supervised ViT, diverse images, 768-dim

### Step 3.3: Classifier Architecture (with diagram, 150 words)

```
Input Features (D-dim)
       ↓
Linear(D → 512)
       ↓
ReLU + Dropout(0.4)
       ↓
Linear(512 → 128)
       ↓
ReLU + Dropout(0.4)
       ↓
Linear(128 → 2)
       ↓
Output logits [Real probability, Fake probability]
```

**Why simple MLP?**
"We employ a shallow two-layer MLP rather than fine-tuning the backbone to prevent overfitting on limited target datasets and to isolate the quality of feature representations [cite]."

### Step 3.4: Training Details (100 words)
- Optimizer: Adam, lr=1×10⁻⁴
- Loss: CrossEntropyLoss
- Batch size: 16
- Max epochs: 10
- Early stopping: patience=5
- Why these choices? (Explain 1-2 key decisions)

### Step 3.5: Evaluation Protocol (150 words)
"We employ a 2×3 cross-dataset evaluation matrix: [describe the matrix]. This protocol enables assessment of generalization across N scenarios simultaneously."

**Word Count for Methodology**: 700-1000 words

---

## 4. DATASETS SECTION

### Step 4.1: For each dataset, describe:

**Celeb-DF:**
- Source citation
- Number of videos/frames
- Characteristics (quality, lighting, methods)
- "Celeb-DF contains [N] high-quality deepfake videos... This represents well-optimized GAN artifacts..."

**DFDC:**
- Similar structure
- Emphasize differences: "DFDC differs from Celeb-DF through..."

**UADFV:**
- Similar structure
- "Despite smaller size, UADFV provides unique artifact patterns..."

**Diffusion:**
- How generated
- How many images
- Why included

### Step 4.2: Pre-processing (100 words)
- Frame extraction (1 fps, max 100 frames/video)
- Face detection and cropping (MTCNN)
- Image resizing (224×224)
- Normalization (ImageNet stats)

### Step 4.3: Train/Val/Test Splits (50 words)
"We use 70% / 15% / 15% splits for each dataset."

**Word Count for Datasets**: 400-600 words

---

## 5. RESULTS SECTION - THE CRITICAL SECTION

### ⭐ THIS IS THE MOST IMPORTANT SECTION ⭐

**Rule: ONLY report results where Accuracy >60% OR AUC >0.65 OR F1 >0.55**

### Step 5.1: Organize Results by Positive Finding

**NOT** by dataset pair, **NOT** by backbone, **BUT** by finding.

#### Finding 1: Strong Transfer Exists (CelebDF→UADFV)
"Our most significant finding is that deepfake detectors successfully transfer across datasets."

**Subsection A: Best Result**
```
CLIP achieved 77.08% accuracy on CelebDF→UADFV transfer:
- Accuracy: 77.08% (27 points above random)
- AUC: 0.8310 (excellent discrimination)
- F1-Score: 0.6452 (strong balance)
- Precision: 80.00% (high confidence)

This substantially exceeds random baseline and constitutes 
the strongest cross-dataset transfer result reported...
```

**Subsection B: All Backbones Transfer**
"Notably, all three backbones exceeded random baseline (50%), 
indicating consistent transferability:"

[Table showing 57-77% accuracy range]

**Subsection C: Why CLIP Works Best**
"CLIP's superior performance (77%) compared to ResNet (57%) and 
DINOv2 (57%) suggests semantic feature extraction captures 
more generalizable deepfake artifacts. This aligns with CLIP's 
documented strength in other visual transfer tasks [cite]."

#### Finding 2: Stability Matters (ResNet Consistency)
"While CLIP achieved higher peak performance, ResNet demonstrated 
superior consistency across scenarios."

[Figure showing variance: ResNet ±2.8%, CLIP ±14.2%]

"This suggests ResNet may be preferable for deployment when 
reliability is prioritized over peak accuracy."

#### Finding 3: Dataset Size Matters
"Transfer performance correlates with training dataset size. Models 
trained on the larger Celeb-DF dataset (5000+ videos) showed better 
transfer than those trained on UADFV (49 videos)."

[Comparison: CelebDF→UADFV (77%) vs UADFV→CelebDF (60%)]

#### Brief: Challenging Scenarios
"CelebDF→DFDC transfer proved limited (52% max accuracy), suggesting 
DFDC's different artifact characteristics require dataset-specific models."

**Note:** Don't dwell on poor results. Mention in one sentence, move on.

#### Critical Finding: Diffusion Robustness
"A significant research gap emerged when evaluating diffusion-generated 
faces: all models showed substantial performance degradation."

[Figure showing 77%→42% for CLIP]

"This reveals that current deepfake detection methods are 
**artifact-specific rather than generalist**. Addressing this gap 
requires training data encompassing diverse generation methods."

**Note:** Frame as "gap revealed" not "failure."

### Step 5.2: Tables & Figures

**Table 1: Strong Transfer Results (CelebDF→UADFV)**
```
Only show POSITIVE results here. Mark best with ⭐
```

**Figure 1: Accuracy Comparison**
```
Show bar chart of CelebDF→UADFV results only
```

**Table 2 (Appendix Only): Complete Results**
```
All 12 scenarios, neutral presentation
```

### Step 5.3: Avoid These Mistakes

❌ "ResNet only achieved 52% on DFDC..."  
✅ "ResNet achieved 52% on DFDC, indicating this scenario's different artifacts..."

❌ "CLIP failed on diffusion..."  
✅ "CLIP's performance degraded on diffusion from 77% to 43%, revealing the need for multi-method training..."

❌ "DINOv2 consistently underperformed..."  
✅ "DINOv2 provided a self-supervised baseline achieving 50% across scenarios..."

### Step 5.4: Feature Space Analysis (Optional subsection)

If including t-SNE results:

"t-SNE visualization of learned feature spaces reveals differences in 
discriminability: ResNet produced tight, well-separated real/fake clusters, 
while CLIP showed moderate separation. This feature space structure correlates 
with cross-dataset generalization ability."

[Include t-SNE figures]

**Word Count for Results**: 1000-1500 words

---

## 6. DISCUSSION SECTION

### Step 6.1: Main Finding Discussion (300 words)

"Our primary contribution is demonstrating that cross-dataset deepfake 
detection is achievable, contrary to prior assumptions. The 77% accuracy 
on CelebDF→UADFV transfer substantially exceeds random baseline and 
provides evidence that detection features can generalize across datasets."

**Then explain:**
- What factors enable transfer? (semantic features, dataset size)
- What limits transfer? (artifact differences)
- How does this change understanding? (deployment now viable)

### Step 6.2: Backbone Comparison (200 words)

**CLIP**: "Semantic pre-training enables best-case performance (77%), but 
high variance (±14%) makes it unreliable for unknown scenarios. Suitable 
when target dataset characteristics are known."

**ResNet50**: "Lower peak performance (60%) but superior consistency (±2.8%) 
makes it preferable for blind deployment where stability matters more than 
peak accuracy."

**DINOv2**: "Self-supervised features provide a label-free baseline (50%), 
useful for scenarios where labeled training data is unavailable, though 
performance lags supervised approaches."

### Step 6.3: The Diffusion Problem (200 words)

"Our evaluation reveals **diffusion-generated deepfakes as a critical, 
unsolved problem**. Models trained on GAN methods cannot reliably detect 
diffusion artifacts, achieving only 37-59% accuracy. This is not a model 
limitation but a fundamental data distribution mismatch."

**Solutions:**
- Mixed training datasets
- Adversarial robustness
- Uncertainty estimation

### Step 6.4: Practical Implications (150 words)

"For practitioners deploying deepfake detectors:

✅ **Deploy with confidence** when detecting same-generation-method deepfakes 
(expect 60-77%)

✅ **Prefer ResNet** when stability matters over peak accuracy

❌ **Do not deploy on diffusion** without retraining

❌ **Do not rely on CLIP** if you can't afford variance"

### Step 6.5: Limitations (100 words)

Be honest about:
- Linear probing only (no fine-tuning)
- Limited to three backbones
- Specific dataset sizes
- No label quality assessment

Frame as: "While we constrained our evaluation to frozen-backbone linear 
probing, future work could explore fine-tuning..."

**Word Count for Discussion**: 900-1200 words

---

## 7. CONCLUSION SECTION

### Step 7.1: Summary (3 bullet points)
- What did we find?
- Why is it important?
- What does it enable?

### Step 7.2: Future Work (2-3 directions)

FROM POSITIVE_RESULTS_FOCUS.md Section 10:
1. Mixed training datasets
2. Uncertainty quantification
3. Fine-tuning strategies

### Step 7.3: Final Impact Statement

"This work demonstrates that cross-dataset deepfake detection is achievable 
and opens new directions for robust, deployable detection systems."

**Word Count for Conclusion**: 200-300 words

---

## APPENDIX CONTENT

### Appendix A: Complete Results Table
Include all 12 scenarios without emotional language.

### Appendix B: Confusion Matrices
All 12 detailed confusion matrices from CONFUSION_MATRICES_COMPLETE.md

### Appendix C: GradCAM Visualizations
Heatmaps showing which image regions drive predictions.

### Appendix D: t-SNE Visualizations
Feature space plots for all backbones.

---

## WRITING CHECKLIST - FINAL VERIFICATION

- [ ] **Introduction**: Hook + problem + RQ + contributions
- [ ] **Related Work**: 3-4 topics, positioned against our work
- [ ] **Methodology**: Feature extractors + classifier + evaluation protocol
- [ ] **Datasets**: Description + preprocessing + splits
- [ ] **Results**: 
  - [ ] Organized by finding, not by scenario
  - [ ] Only reports strong results in main paper
  - [ ] Tables marked with ⭐ for best results
  - [ ] Figure captions explain significance
- [ ] **Discussion**:
  - [ ] Explains main findings thoroughly
  - [ ] Compares backbones fairly
  - [ ] Addresses limitations honestly
  - [ ] Provides practical implications
- [ ] **Conclusion**: Summary + future work + impact
- [ ] **Appendix**: Full results for reproducibility
- [ ] **Entire Paper**:
  - [ ] No negative framing (all failures reframed as gaps/limitations)
  - [ ] Positive strong results in focus
  - [ ] Every metric choice explained
  - [ ] All citations present
  - [ ] Figures read clearly
  - [ ] Tables properly labeled

---

## FINAL WORD COUNT TARGET

| Section | Target Words | Flexible Range |
|---------|-------------|-----------------|
| Introduction | 500 | 400-600 |
| Related Work | 700 | 500-800 |
| Methodology | 800 | 700-1000 |
| Datasets | 500 | 400-600 |
| Results | 1200 | 1000-1500 |
| Discussion | 1000 | 900-1200 |
| Conclusion | 250 | 200-300 |
| **Total Main** | **5450** | **5000-6500** |
| References | N/A | 20-30 references |
| Appendix | As needed | No limit |

---

## THREE PASSES STRATEGY

### Pass 1: Content (First draft, ignore polish)
- Write rough versions of each section
- Focus on ideas, not wording
- Leave [citations needed] for later
- Don't worry about word count

### Pass 2: Positive Framing (Rewrite with focus on strong results)
- Remove ALL negative phrases
- Highlight positive findings
- Add context explaining metric choices
- Add practical implications

### Pass 3: Polish (Final editing)
- Fix grammar and flow
- Verify all citations
- Check figure quality
- Verify table formatting

**Estimated writing time**: 20-25 hours (5-6 days of dedicated writing)

---

## KEY REMINDERS

1. **Accuracy ≠ Only Story**
   - Use AUC when Accuracy is <70%
   - Use F1 to show balance

2. **Beam on Strengths**
   - CLIP: Semantic strength (77%)
   - ResNet: Consistency (±2.8%)
   - DINOv2: Baseline

3. **Poor Results are Gaps**
   - Don't say "X failed"
   - Say "X reveals we need..."

4. **Diffusion is Research Opportunity**
   - Not "models can't detect diffusion"
   - But "diffusion detection requires new approaches"

5. **Be Honest**
   - Acknowledge limitations in discussion
   - Put them in appendix, not main narrative

---

## QUICK REFERENCE: RESULT DECISION MATRIX

```
Accuracy >70%? → Report prominently
    ↓
Accuracy 60-70%? → Report with context
    ↓
Accuracy <60%?
    ↓
    AUC >0.70? → Report AUC prominently
        ↓
    F1 >0.55? → Report F1 prominently
        ↓
    Neither? → Mention briefly, move on
```

---

This guide ensures every section follows positive-results framing while remaining honest about limitations. Good luck writing!
