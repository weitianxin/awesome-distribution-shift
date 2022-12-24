# awesome-distribution-shift ![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) ![Awesome](https://awesome.re/badge.svg)

A curated list of papers and resources about the distribution shift in machine learning. I categorize them based on their topic and content. I will try to make this list updated.

Here is an example of distribution shift in images across domains from [DomainBed](https://github.com/facebookresearch/DomainBed).

![avatar](https://github.com/weitianxin/awesome-distribution-shift/blob/main/example.png)

I categorize the papers on distribution shift as follows. If you found any error or any missed paper, please don't hesitate to add.

Continuously updated

## Benchmark
[ICLR 2021] **In Search of Lost Domain Generalization** [[paper]](https://arxiv.org/abs/2007.01434) [[code (DomainBed)]](https://github.com/facebookresearch/DomainBed)

[ICLR 2021] **BREEDS: Benchmarks for Subpopulation Shift** [[paper]](https://arxiv.org/abs/2008.04859) [[code]](https://github.com/MadryLab/BREEDS-Benchmarks)

[ICML 2021] **WILDS: A Benchmark of in-the-Wild Distribution Shifts** [[paper]](https://arxiv.org/abs/2012.07421) [[code]](https://github.com/p-lambda/wilds)

[NeurIPS 2021] **Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks** [[paper]](https://openreview.net/pdf?id=qM45LHaWM6E) [[code]](https://github.com/Shifts-Project/shifts)

[ICLR 2022] **MetaShift: A Dataset of Datasets for Evaluating Contextual Distribution Shifts and Training Conflicts** [[paper]](https://openreview.net/pdf?id=MTex8qKavoS) [[code]](https://metashift.readthedocs.io/en/latest/)

[NeurIPS 2022] **GOOD: A Graph Out-of-Distribution Benchmark** [[paper]](https://arxiv.org/abs/2206.08452) [[code]](https://github.com/divelab/GOOD)

[NeurIPS 2022] **BOND: Benchmarking Unsupervised Outlier Node Detection on Static Attributed Graphs** [[paper]](https://arxiv.org/abs/2206.10071) [[code]](https://github.com/pygod-team/pygod/tree/main/benchmark)

[NeurIPS 2022] **ADBench: Anomaly Detection Benchmark** [[paper]](https://arxiv.org/abs/2206.09426) [[code]](https://github.com/Minqi824/ADBench)

[NeurIPS 2022] **Wild-Time: A Benchmark of in-the-Wild Distribution Shifts over Time** [[paper]](https://arxiv.org/abs/2211.14238) [[code]](https://github.com/huaxiuyao/Wild-Time)

[NeurIPS 2022] **OpenOOD: Benchmarking Generalized Out-of-Distribution Detection** [[paper]](https://openreview.net/pdf?id=gT6j4_tskUt) [[code]](https://github.com/Jingkang50/OpenOOD)

[NeurIPS 2022] **AnoShift: A Distribution Shift Benchmark for Unsupervised Anomaly Detection** [[paper]](https://arxiv.org/abs/2206.15476) [[code]](https://github.com/bit-ml/anoshift)

[CVPR 2022] **OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization** [[paper]](https://arxiv.org/abs/2106.03721) [[code]](https://github.com/m-Just/OoD-Bench)

[CVPR 2022] **The Auto Arborist Dataset: A Large-Scale Benchmark for Multiview Urban Forest Monitoring Under Domain Shift** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Beery_The_Auto_Arborist_Dataset_A_Large-Scale_Benchmark_for_Multiview_Urban_CVPR_2022_paper.pdf) [[code]](https://google.github.io/auto-arborist/)

[ECCV 2022] **OOD-CV: A Benchmark for Robustness to Out-of-Distribution Shifts of Individual Nuisances in Natural Images** [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680158.pdf) [[code]](https://github.com/eccv22-ood-workshop/ROBIN-dataset)

[arxiv 2022] **DrugOOD: OOD Dataset Curator and Benchmark for AI Aided Drug Discovery** [[paper]](https://arxiv.org/abs/2201.09637) [[code]](https://github.com/tencent-ailab/DrugOOD)



## Generalization

### Domain Generalization & Out-of-distribution Robustness
There are mainly two types of distribution shift: domain shift (testing on unseen domains) and subpopulation shift (the domains of testing data are seen but underrepresented in the training data). Below figure from [GOOD](https://github.com/divelab/GOOD) well demonstrates them.
![avatar](https://github.com/weitianxin/awesome-distribution-shift/blob/main/dis%20shift.png)
**Domain Generalization** mainly studies **domain shift**, while **Out-of-distribution Robustness** studies **both** of them.

#### Domain Generalization
Learning to Generalize: Meta-Learning for Domain Generalization
Domain Generalization With Adversarial Feature Learning
Deep Domain Generalization via Conditional Invariant Adversarial Networks
MetaReg: Towards Domain Generalization Using Meta-Regularization

Domain Generalization via Entropy Regularization
Learning to Optimize Domain Specific Normalization for Domain Generalization
Learning From Extrinsic and Intrinsic Supervisions for Domain Generalization
Learning to Balance Specificity and Invariance for in and Out of Domain Generalization
Learning to Learn Single Domain Generalization
Domain Generalization With Optimal Transport and Metric Learning
Learning to Generate Novel Domains for Domain Generalization

Self-Challenging Improves Cross-Domain Generalization
Domain Generalization By Marginal Transfer Learning
Swad: Domain Generalization By Seeking Flat Minima
Selfreg: Self-Supervised Contrastive Regularization for Domain Generalization
A Simple Feature Augmentation for Domain Generalization

Gradient Matching for Domain Generalization
PCL: Proxy-Based Contrastive Learning for Domain Generalization
Compound Domain Generalization Via Meta-Knowledge Encoding
Domain Generalization By Mutual-Information Regularization With Pre-Trained Models
Dna: Domain Generalization With Diversified Neural Averaging

Towards Unsupervised Domain Generalization
Unsupervised Domain Generalization By Learning a Bridge Across Domains

#### Out-of-distribution Robustness
Invariant risk minimization

Distributionally robust neural networks for group shifts: on the importance of regularization for worst-case generalization
Out-of-Distribution Generalization via Risk Extrapolation (REx)

Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization
In-N-Out: pre-training and self-training using auxiliary information for out-of-distribution robustness
Accuracy on the line: on the strong correlation between out-of-distribution and in-distribution generalization

Improving Out-of-Distribution Robustness via Selective Augmentation
Fishr: Invariant Gradient Variances for Out-of-distribution Generalization
Diverse Weight Averaging for Out-of-Distribution Generalization

Just Train Twice: Improving Group Robustness without Training Group Information
Model agnostic sample reweighting for out-of-distribution learning
Correct-N-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations


### Domain Adaptation
Connect, not collapse: explaining contrastive learning for unsupervised domain adaptation

### Test-time Adaptation


## Modality
Above papers study the distribution shift on images. There are also many applications to other data modalities.
### Graph

### Text

### Time Series

### Video

### Speech

### Tabular Data

## Decentralized (Federated)
Besides generalization of centralized learning, transferability of decentralized setting (Federated Learning) is also studied.

## Detection
Besides generalization, detection, fairness and robustness are also studied.

## Fairness

## Robustness

## Learning Strategy





