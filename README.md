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
**Domain Generalization** mainly studies **domain shift**, while **Out-of-distribution Robustness** studies **both** of them. These two research directions are very related and share a lot in common.

#### Out-of-distribution Robustness
Invariant Risk Minimization

Distributionally Robust Neural Networks for Group Shifts: on the Importance of Regularization for Worst-Case Generalization

Out-of-Distribution Generalization Via Risk Extrapolation (REx)

Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization

In-N-Out: Pre-Training and Self-Training Using Auxiliary Information for Out-of-Distribution Robustness

Accuracy on the Line: on the Strong Correlation Between Out-of-Distribution and in-Distribution Generalization

Improving Out-of-Distribution Robustness Via Selective Augmentation

Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization

Diverse Weight Averaging for Out-of-Distribution Generalization

##### w/o group label
Just Train Twice: Improving Group Robustness Without Training Group Information

Model Agnostic Sample Reweighting for Out-of-Distribution Learning

Correct-N-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations

#### Domain Generalization
Learning to Generalize: Meta-Learning for Domain Generalization

Domain Generalization With Adversarial Feature Learning

Deep Domain Generalization Via Conditional Invariant Adversarial Networks

MetaReg: Towards Domain Generalization Using Meta-Regularization

Domain Generalization Via Entropy Regularization

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

Domain Generalization Without Excess Empirical Risk

Towards Unsupervised Domain Generalization

Unsupervised Domain Generalization By Learning a Bridge Across Domains

### Domain Adaptation
Unsupervised domain adaptation by backpropagation

Domain-Adversarial Training of Neural Networks

Correlation Alignment for Unsupervised Domain Adaptation

Unsupervised domain adaptation with residual transfer networks

Learning transferrable representations for unsupervised domain adaptation

Return of frustratingly easy domain adaptation

Adversarial discriminative domain adaptation

Balanced distribution adaptation for transfer learning

Multi-adversarial domain adaptation

Universal domain adaptation

Contrastive adaptation network for unsupervised domain adaptation

On learning invariant representations for domain adaptation

Visual domain adaptation with manifold embedded distribution alignment

Reliable weighted optimal transport for unsupervised domain adaptation

Universal source-free domain adaptation

Dacs: Domain adaptation via cross-domain mixed sampling

Dynamic weighted learning for unsupervised domain adaptation

Generalized source-free domain adaptation

Adaptive adversarial network for source-free domain adaptation

Connect, not collapse: explaining contrastive learning for unsupervised domain adaptation

### Test-time Adaptation/Training
Tent: Fully test-time adaptation by entropy minimization

Test-time training with self-supervision for generalization under distribution shifts

Mt3: Meta test-time training for self-supervised test-time adaption

Contrastive Test-Time Adaptation

Test-time unsupervised domain adaptation

Test-time adaptation to distribution shift by confidence maximization and input transformation

Parameter-free Online Test-time Adaptation

Test-Time Adaptation via Conjugate Pseudo-labels

Efficient Test-Time Model Adaptation without Forgetting

TTT++: When Does Self-supervised Test-time Training Fail or Thrive?

Test-time training with masked autoencoders


## Modality






the distribution shift on images. There are also many applications to other data modalities.
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





