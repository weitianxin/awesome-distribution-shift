# awesome-distribution-shift ![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) ![Awesome](https://awesome.re/badge.svg)

A curated list of papers and resources about the distribution shift in machine learning. I categorize them based on their topic and content. I will try to make this list updated.

Here is an example of distribution shift in images across domains from [DomainBed](https://github.com/facebookresearch/DomainBed).

![avatar](https://github.com/weitianxin/awesome-distribution-shift/blob/main/figs/example.png)

I categorize the papers on distribution shift as follows. If you found any error or any missed paper, please don't hesitate to add.

To be updated

## Contents
- [awesome-distribution-shift  ](#awesome-distribution-shift--)
  - [Contents](#contents)
  - [Benchmark](#benchmark)
  - [Generalization](#generalization)
    - [Domain Generalization \& Out-of-distribution Robustness](#domain-generalization--out-of-distribution-robustness)
      - [Out-of-distribution Robustness](#out-of-distribution-robustness)
      - [Domain Generalization](#domain-generalization)
    - [Domain Adaptation](#domain-adaptation)
    - [Test-time Adaptation/Training](#test-time-adaptationtraining)
  - [Data Modality](#data-modality)
    - [Graph](#graph)
    - [Text](#text)
    - [Time Series](#time-series)
    - [Video](#video)
    - [Speech](#speech)
    - [Tabular Data](#tabular-data)
    - [Others (RecSys)](#others-recsys)
  - [Decentralized (Federated)](#decentralized-federated)
  - [Detection](#detection)
  - [Fairness](#fairness)
  - [Robustness](#robustness)
  - [Learning Strategy](#learning-strategy)

## Benchmark

- [ICLR 2021] **In Search of Lost Domain Generalization** [[paper]](https://arxiv.org/abs/2007.01434) [[code (DomainBed)]](https://github.com/facebookresearch/DomainBed)
- [ICLR 2021] **BREEDS: Benchmarks for Subpopulation Shift** [[paper]](https://arxiv.org/abs/2008.04859) [[code]](https://github.com/MadryLab/BREEDS-Benchmarks)
- [ICML 2021] **WILDS: A Benchmark of in-the-Wild Distribution Shifts** [[paper]](https://arxiv.org/abs/2012.07421) [[code]](https://github.com/p-lambda/wilds)
- [NeurIPS 2021] **Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks** [[paper]](https://openreview.net/pdf?id=qM45LHaWM6E) [[code]](https://github.com/Shifts-Project/shifts)
- [ICLR 2022] **MetaShift: A Dataset of Datasets for Evaluating Contextual Distribution Shifts and Training Conflicts** [[paper]](https://openreview.net/pdf?id=MTex8qKavoS) [[code]](https://metashift.readthedocs.io/en/latest/)
- [NeurIPS 2022] **GOOD: A Graph Out-of-Distribution Benchmark** [[paper]](https://arxiv.org/abs/2206.08452) [[code]](https://github.com/divelab/GOOD)
- [NeurIPS 2022] **BOND: Benchmarking Unsupervised Outlier Node Detection on Static Attributed Graphs** [[paper]](https://arxiv.org/abs/2206.10071) [[code]](https://github.com/pygod-team/pygod/tree/main/benchmark)
- [NeurIPS 2022] **ADBench: Anomaly Detection Benchmark** [[paper]](https://arxiv.org/abs/2206.09426) [[code]](https://github.com/Minqi824/ADBench)
- [NeurIPS 2022] **Wild-Time: A Benchmark of in-the-Wild Distribution Shifts over Time** [[paper]](https://arxiv.org/abs/2211.14238) [[code]](https://github.com/huaxiuyao/Wild-Time)
- [NeurIPS 2022] **OpenOOD: Benchmarking Generalized Out-of-Distribution Detection** [[paper]](https://openreview.net/pdf?id=gT6j4_tskUt) [[code]](https://github.com/Jingkang50/OpenOOD)
- [NeurIPS 2022] **AnoShift: A Distribution Shift Benchmark for Unsupervised Anomaly Detection** [[paper]](https://arxiv.org/abs/2206.15476) [[code]](https://github.com/bit-ml/anoshift)
- [CVPR 2022] **OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization** [[paper]](https://arxiv.org/abs/2106.03721) [[code]](https://github.com/m-Just/OoD-Bench)
- [CVPR 2022] **The Auto Arborist Dataset: A Large-Scale Benchmark for Multiview Urban Forest Monitoring Under Domain Shift** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Beery_The_Auto_Arborist_Dataset_A_Large-Scale_Benchmark_for_Multiview_Urban_CVPR_2022_paper.pdf) [[code]](https://google.github.io/auto-arborist/)
- [ECCV 2022] **OOD-CV: A Benchmark for Robustness to Out-of-Distribution Shifts of Individual Nuisances in Natural Images** [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680158.pdf) [[code]](https://github.com/eccv22-ood-workshop/ROBIN-dataset)
- [arxiv 2022] **DrugOOD: OOD Dataset Curator and Benchmark for AI Aided Drug Discovery** [[paper]](https://arxiv.org/abs/2201.09637) [[code]](https://github.com/tencent-ailab/DrugOOD)

## Generalization

### Domain Generalization & Out-of-distribution Robustness

There are mainly two types of distribution shift: domain shift (testing on unseen domains) and subpopulation shift (the domains of testing data are seen but underrepresented in the training data). Below figure from [GOOD](https://github.com/divelab/GOOD) well demonstrates them.
![avatar](https://github.com/weitianxin/awesome-distribution-shift/blob/main/figs/dis%20shift.png)
**Domain Generalization** mainly studies **domain shift**, while **Out-of-distribution Robustness** studies **both** of them. These two research directions are very related and share a lot in common.

#### Out-of-distribution Robustness

- [arxiv 2019] **Invariant Risk Minimization** [[paper]](https://arxiv.org/abs/1907.02893)
- [ICLR 2020] **Distributionally Robust Neural Networks for Group Shifts: on the Importance of Regularization for Worst-Case Generalization** [[paper]](https://openreview.net/pdf?id=ryxGuJrFvS)
- [ICLR 2021] **In-N-Out: Pre-Training and Self-Training Using Auxiliary Information for Out-of-Distribution Robustness** [[paper]](https://arxiv.org/pdf/2012.04550)
- [ICML 2021] **Out-of-Distribution Generalization Via Risk Extrapolation (REx)** [[paper]](http://proceedings.mlr.press/v139/krueger21a/krueger21a.pdf)
- [ICML 2021] **Accuracy on the Line: on the Strong Correlation Between Out-of-Distribution and in-Distribution Generalization** [[paper]](https://proceedings.mlr.press/v139/miller21b/miller21b.pdf)
- [NeurIPS 2021] **Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization** [[paper]](https://openreview.net/pdf?id=jlchsFOLfeF)
- [ICML 2022] **Improving Out-of-Distribution Robustness Via Selective Augmentation** [[paper]](https://arxiv.org/pdf/2201.00299)
- [ICML 2022] **Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization** [[paper]](https://arxiv.org/pdf/2109.02934)
- [NeurIPS 2022] **Diverse Weight Averaging for Out-of-Distribution Generalization** [[paper]](https://arxiv.org/pdf/2205.09739)

**W/o group label**

- [ICML 2021] **Just Train Twice: Improving Group Robustness Without Training Group Information** [[paper]](https://arxiv.org/pdf/2107.09044)
- [ICML 2022] **Model Agnostic Sample Reweighting for Out-of-Distribution Learning** [[paper]](https://proceedings.mlr.press/v162/zhou22d/zhou22d.pdf)
- [ICML 2022] **Correct-N-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations** [[paper]](https://arxiv.org/pdf/2203.01517)

#### Domain Generalization

- [AAAI 2018] **Learning to Generalize: Meta-Learning for Domain Generalization** [[paper]](https://arxiv.org/pdf/1710.03463)
- [CVPR 2018] **Domain Generalization With Adversarial Feature Learning** [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf)
- [ECCV 2018] **Deep Domain Generalization Via Conditional Invariant Adversarial Networks** [[paper]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf)
- [NeurIPS 2018] **MetaReg: Towards Domain Generalization Using Meta-Regularization** [[paper]](https://papers.nips.cc/paper/2018/file/647bba344396e7c8170902bcf2e15551-Paper.pdf)
- [NeurIPS 2019] **Domain Generalization Via Model-Agnostic Learning of Semantic Features** [[paper]](https://proceedings.neurips.cc/paper/2019/file/2974788b53f73e7950e8aa49f3a306db-Paper.pdf)
- [CVPR 2019] **Episodic Training for Domain Generalization** [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Episodic_Training_for_Domain_Generalization_ICCV_2019_paper.pdf)
- [CVPR 2019] **Domain Generalization By Solving Jigsaw Puzzles** [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.pdf)
- [NeurIPS 2020] **Domain Generalization Via Entropy Regularization** [[paper]](https://proceedings.neurips.cc/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Paper.pdf)
- [ECCV 2020] **Learning to Optimize Domain Specific Normalization for Domain Generalization** [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670069.pdf)
- [ECCV 2020] **Learning From Extrinsic and Intrinsic Supervisions for Domain Generalization** [[paper]](https://arxiv.org/pdf/2007.09316)
- [ECCV 2020] **Learning to Balance Specificity and Invariance for in and Out of Domain Generalization** [[paper]](https://arxiv.org/pdf/2008.12839)
- [ECCV 2020] **Self-Challenging Improves Cross-Domain Generalization** [[paper]](https://arxiv.org/pdf/2007.02454)
- [ECCV 2020] **Learning to Generate Novel Domains for Domain Generalization** [[paper]](https://arxiv.org/pdf/2007.03304)
- [CVPR 2020] **Learning to Learn Single Domain Generalization** [[paper]](https://arxiv.org/pdf/2003.13216)
- [JMLR 2021] **Domain Generalization By Marginal Transfer Learning** [[paper]](https://www.jmlr.org/papers/volume22/17-679/17-679.pdf)
- [NeurIPS 2021] **Swad: Domain Generalization By Seeking Flat Minima** [[paper]](https://proceedings.neurips.cc/paper/2021/file/bcb41ccdc4363c6848a1d760f26c28a0-Paper.pdf)
- [NeurIPS 2021] **Model-based Domain Generalization** [[paper]](https://proceedings.neurips.cc/paper/2021/file/a8f12d9486cbcc2fe0cfc5352011ad35-Paper.pdf)
- [ICCV 2021] **Selfreg: Self-Supervised Contrastive Regularization for Domain Generalization** [[paper]](http://openaccess.thecvf.com/content/ICCV2021/papers/Kim_SelfReg_Self-Supervised_Contrastive_Regularization_for_Domain_Generalization_ICCV_2021_paper.pdf)
- [ICCV 2021] **A Simple Feature Augmentation for Domain Generalization** [[paper]](http://openaccess.thecvf.com/content/ICCV2021/papers/Li_A_Simple_Feature_Augmentation_for_Domain_Generalization_ICCV_2021_paper.pdf)
- [ICLR 2022] **Gradient Matching for Domain Generalization** [[paper]](https://arxiv.org/pdf/2104.09937)
- [CVPR 2022] **PCL: Proxy-Based Contrastive Learning for Domain Generalization** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Yao_PCL_Proxy-Based_Contrastive_Learning_for_Domain_Generalization_CVPR_2022_paper.pdf)
- [ECCV 2022] **Domain Generalization By Mutual-Information Regularization With Pre-Trained Models** [[paper]](https://arxiv.org/pdf/2203.10789)
- [ICML 2022] **Dna: Domain Generalization With Diversified Neural Averaging** [[paper]](https://proceedings.mlr.press/v162/chu22a/chu22a.pdf)
- [NeurIPS 2022] **Domain Generalization Without Excess Empirical Risk** [[paper]](https://openreview.net/pdf?id=pluyPFTiTeJ)

**W/o domain label**

Explicitly introduce compound domain generalization without domain labels (some previous work also don't rely on domain labels)

- [CVPR 2022] **Compound Domain Generalization Via Meta-Knowledge Encoding** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Compound_Domain_Generalization_via_Meta-Knowledge_Encoding_CVPR_2022_paper.pdf)

**Unsupervised**

- [CVPR 2022] **Towards Unsupervised Domain Generalization** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Towards_Unsupervised_Domain_Generalization_CVPR_2022_paper.pdf)
- [CVPR 2022] **Unsupervised Domain Generalization By Learning a Bridge Across Domains** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Harary_Unsupervised_Domain_Generalization_by_Learning_a_Bridge_Across_Domains_CVPR_2022_paper.pdf)

### Domain Adaptation

In the setting of domain adaptation, the model can access partial target domain data (usually unsupervised). There are also many variants like open-set, evolving, dynamic domain adaptation, etc.

- [ICML 2015] **Unsupervised Domain Adaptation By Backpropagation** [[paper]](http://proceedings.mlr.press/v37/ganin15.pdf)
- [JMLR 2016] **Domain-Adversarial Training of Neural Networks** [[paper]](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf)
- [ECCV 2016] **Deep Coral: Correlation Alignment for Deep Domain Adaptation** [[paper]](https://arxiv.org/abs/1607.01719)
- [NeurIPS 2016] **Unsupervised Domain Adaptation With Residual Transfer Networks** [[paper]](https://proceedings.neurips.cc/paper/2016/file/ac627ab1ccbdb62ec96e702f07f6425b-Paper.pdf)
- [NeurIPS 2016] **Learning Transferrable Representations for Unsupervised Domain Adaptation** [[paper]](https://proceedings.neurips.cc/paper/2016/file/b59c67bf196a4758191e42f76670ceba-Paper.pdf)
- [CVPR 2017] **Adversarial Discriminative Domain Adaptation** [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf)
- [ICDM 2017] **Balanced Distribution Adaptation for Transfer Learning** [[paper]](https://ieeexplore.ieee.org/iel7/8211002/8215462/08215613.pdf?casa_token=Kdomni1I_-YAAAAA:wg0xxHwxoJruSejuzmpfe66So7_WSL2Lr1yHa4OfdLVfhXSMxaqzDmTQwPEZkbmi2ggx_Ymqm4Nl)
- [ECCV 2018] **Open set domain adaptation by backpropagation** [[paper]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kuniaki_Saito_Adversarial_Open_Set_ECCV_2018_paper.pdf)
- [AAAI 2018] **Multi-Adversarial Domain Adaptation** [[paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17067/16644)
- [MM 2018] **Visual Domain Adaptation With Manifold Embedded Distribution Alignment** [[paper]](https://dl.acm.org/doi/pdf/10.1145/3240508.3240512)
- [CVPR 2019] **Universal Domain Adaptation** [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/You_Universal_Domain_Adaptation_CVPR_2019_paper.pdf)
- [CVPR 2019] **Contrastive Adaptation Network for Unsupervised Domain Adaptation** [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)
- [ICML 2019] **On Learning Invariant Representations for Domain Adaptation** [[paper]](http://proceedings.mlr.press/v97/zhao19a/zhao19a.pdf)
- [CVPR 2020] **Reliable Weighted Optimal Transport for Unsupervised Domain Adaptation** [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf)
- [ICML 2020] **Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation** [[paper]](http://proceedings.mlr.press/v119/liang20a/liang20a.pdf)
- [TPAMI 2020] **Maximum Density Divergence for Domain Adaptation** [[paper]](https://ieeexplore.ieee.org/iel7/34/4359286/09080115.pdf?casa_token=DkacOMonv3gAAAAA:4DJ1THgujrfZcz3ITdd09Xo_RwkIp6RKS3-Fu2-ySg_I7xeIAFKeGL_GVl6zchVZTS2xq8LasuSk)
- [AAAI 2020] **Adversarial Domain Adaptation with Domain Mixup** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/6123)
- [WACV 2021] **Dacs: Domain Adaptation Via Cross-Domain Mixed Sampling** [[paper]](https://openaccess.thecvf.com/content/WACV2021/papers/Tranheden_DACS_Domain_Adaptation_via_Cross-Domain_Mixed_Sampling_WACV_2021_paper.pdf)
- [CVPR 2021] **Dynamic Weighted Learning for Unsupervised Domain Adaptation** [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Xiao_Dynamic_Weighted_Learning_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf)
- [ICML 2022] **Connect, Not Collapse: Explaining Contrastive Learning for Unsupervised Domain Adaptation** [[paper]](https://proceedings.mlr.press/v162/shen22d/shen22d.pdf)
- [TPAMI 2022] **Cross-domain Contrastive Learning for Unsupervised Domain Adaptation** [[paper]](https://ieeexplore.ieee.org/iel7/6046/4456689/09695359.pdf?casa_token=BQ17etjoVUwAAAAA:pSLouzFWW4fQm_klgNughmr8NxIojk1tPFvmV5p271abcjn9MSzC_6p_i6JAWX0S3rEy9fIF5_iK)
- [WACV 2022] **Federated Multi-target Domain Adaptation** [[paper]](https://openaccess.thecvf.com/content/WACV2022/papers/Yao_Federated_Multi-Target_Domain_Adaptation_WACV_2022_paper.pdf)

### Test-time Adaptation/Training

In the scenario of test-time adaptation, the model is pre-trained on the source domain data and is adapted during the testing on the target domain.

- [ICML 2020] **Test-time training with self-supervision for generalization under distribution shifts** [[paper]](http://proceedings.mlr.press/v119/sun20b/sun20b.pdf)
- [MICCAI 2020] **Test-time Unsupervised Domain Adaptation** [[paper]](https://arxiv.org/pdf/2010.01926)
- [ICLR 2021] **Tent: Fully test-time adaptation by entropy minimization** [[paper]](https://arxiv.org/pdf/2006.10726)
- [NeurIPS 2021] **TTT++: When Does Self-supervised Test-time Training Fail or Thrive?** [[paper]](https://proceedings.neurips.cc/paper/2021/file/b618c3210e934362ac261db280128c22-Paper.pdf)
- [ICML 2022] **Mt3: Meta Test-time Training for Self-supervised Test-time Adaption** [[paper]](https://proceedings.mlr.press/v151/bartler22a/bartler22a.pdf)
- [CVPR 2022] **Contrastive Test-Time Adaptation** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Contrastive_Test-Time_Adaptation_CVPR_2022_paper.pdf)
- [CVPR 2022] **Parameter-free Online Test-time Adaptation** [[paper]](http://openaccess.thecvf.com/content/CVPR2022/papers/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.pdf)
- [NeurIPS 2022] **Test-Time Adaptation via Conjugate Pseudo-labels** [[paper]](https://arxiv.org/pdf/2207.09640)
- [ICML 2022] **Efficient Test-Time Model Adaptation without Forgetting** [[paper]](https://arxiv.org/pdf/2204.02610)
- [NeurIPS 2022] **Test-time Training with Masked Autoencoders** [[paper]](https://arxiv.org/pdf/2209.07522)
- [NeurIPS 2022] **Meta-DMoE: Adapting to Domain Shift by Meta-Distillation from Mixture-of-Experts** [[paper]](https://arxiv.org/pdf/2210.03885)

Also known as source-free domain adaptation
- [CVPR 2020] **Universal Source-Free Domain Adaptation** [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Kundu_Universal_Source-Free_Domain_Adaptation_CVPR_2020_paper.pdf)
- [CVPR 2020] **Model adaptation: Unsupervised domain adaptation without source data** [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Model_Adaptation_Unsupervised_Domain_Adaptation_Without_Source_Data_CVPR_2020_paper.pdf)
- [ICCV 2021] **Generalized Source-Free Domain Adaptation** [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Generalized_Source-Free_Domain_Adaptation_ICCV_2021_paper.pdf)
- [ICCV 2021] **Adaptive Adversarial Network for Source-Free Domain Adaptation** [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xia_Adaptive_Adversarial_Network_for_Source-Free_Domain_Adaptation_ICCV_2021_paper.pdf)
- [CVPR 2022] **Source-Free Domain Adaptation via Distribution Estimation** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Source-Free_Domain_Adaptation_via_Distribution_Estimation_CVPR_2022_paper.pdf)
- [ICML 2022] **Balancing Discriminability and Transferability for Source-Free Domain Adaptation** [[paper]](https://arxiv.org/abs/2206.08009?context=cs.LG)
- [NeurIPS 2022] **Attracting and Dispersing: A Simple Approach for Source-free Domain Adaptation** [[paper]](https://openreview.net/forum?id=ZlCpRiZN7n)
- [NeurIPS 2022] **Divide and Contrast: Source-free Domain Adaptation via Adaptive Contrastive Learning** [[paper]](https://openreview.net/forum?id=NjImFaBEHl)
## Data Modality

Above papers study the distribution shift on images. There are also many applications to other data modalities.

### Graph
- [ICLR 2022] **Handling Distribution Shifts on Graphs: An Invariance Perspective** [[paper]](https://arxiv.org/pdf/2202.02466)
- [ICLR 2022] **Discovering Invariant Rationales for Graph Neural Networks** [[paper]](https://arxiv.org/pdf/2201.12872)
- [ICML 2022] **Interpretable and Generalizable Graph Learning Via Stochastic Attention Mechanism** [[paper]](https://proceedings.mlr.press/v162/miao22a/miao22a.pdf)
- [KDD 2022] **Causal Attention for Interpretable and Generalizable Graph Classification** [[paper]](https://dl.acm.org/doi/pdf/10.1145/3534678.3539366?casa_token=zKvdUqGb124AAAAA:Rg8bhY1lJeqahX1PHEVUs7UbtWDjuihnvfbEfOvTgBt6MO1rAF5y3L7sCj0CbJTMN8dLvGFkFzWPK_w)
- [NeurIPS 2022] **Dynamic Graph Neural Networks Under Spatio-temporal Distribution Shift** [[paper]](https://openreview.net/pdf?id=1tIUqrUuJxx)
- [NeurIPS 2022] **Learning Substructure Invariance for Out-of-Distribution Molecular Representations** [[paper]](https://openreview.net/pdf?id=2nWUNTnFijm)
- [NeurIPS 2022] **Learning Invariant Graph Representations for Out-of-Distribution Generalization** [[paper]](https://openreview.net/pdf?id=acKK8MQe2xc)
- [NeurIPS 2022] **Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs** [[paper]](https://openreview.net/pdf?id=A6AFK_JwrIW)
- [WSDM 2023] **Alleviating Structural Distribution Shift in Graph Anomaly Detection** [[paper]](http://staff.ustc.edu.cn/~hexn/papers/wsdm23-GDN.pdf)

Test-time adaptation
- [arxiv 2022] **Test-Time Training for Graph Neural Networks** [[paper]](https://arxiv.org/pdf/2210.08813)
- [arxiv 2022] **Empowering Graph Representation Learning with Test-time Graph Transformation** [[paper]](https://arxiv.org/pdf/2210.03561)

### Text
Many applications in different NLP tasks.
- [CVPR 2019] **Sequence-to-sequence Domain Adaptation Network for Robust Text Image Recognition** [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Sequence-To-Sequence_Domain_Adaptation_Network_for_Robust_Text_Image_Recognition_CVPR_2019_paper.pdf)
- [NAACL 2019] **Overcoming Catastrophic Forgetting During Domain Adaptation of Neural Machine Translation** [[paper]](https://aclanthology.org/N19-1209/)
- [AAAI 2020] **Multi-source Domain Adaptation for Text Classification Via Distancenet-bandits** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/download/6288/6144)
- [COLING 2020] **Semi-supervised Domain Adaptation for Dependency Parsing via Improved Contextualized Word Representations** [[paper]](https://aclanthology.org/2020.coling-main.338/)
- [ACL 2020] **Pretrained Transformers Improve Out-of-distribution Robustness** [[paper]](https://arxiv.org/pdf/2004.06100)
- [EMNLP 2021] **Contrastive Domain Adaptation for Question Answering Using Limited Text Corpora** [[paper]](https://arxiv.org/pdf/2108.13854)
- [EMNLP 2021] **Pdaln: Progressive Domain Adaptation over a Pre-trained Model for Low-resource Cross-domain Named Entity Recognition** [[paper]](https://aclanthology.org/2021.emnlp-main.442/)
- [ACL 2021] **Matching Distributions Between Model and Data: Cross-domain Knowledge Distillation for Unsupervised Domain Adaptation** [[paper]](https://aclanthology.org/2021.acl-long.421/)
- [ACL 2021] **Bridge-based Active Domain Adaptation for Aspect Term Extraction** [[paper]](https://aclanthology.org/2021.acl-long.27/)
- [ECCV 2022] **Grounding Visual Representations with Texts for Domain Generalization** [[paper]](https://arxiv.org/pdf/2207.10285)
- [KAIS 2022] **Knowledge distillation for bert unsupervised domain adaptation** [[paper]](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/article/10.1007/s10115-022-01736-y&casa_token=VgVxkeLR5wcAAAAA:ujMylYAlDW-elyTA6SHGl2L0zl_M32WSZIiwLSSwKfsBnUrJ6aaLJRhLx9U-lD1Wxc_3ty2tZzwsp-fe7jc)
- [ACL 2022] **Semi-supervised Domain Adaptation for Dependency Parsing with Dynamic Matching Network** [[paper]](https://aclanthology.org/2022.acl-long.74/)
...


### Time Series
- [ICLR 2017] **Variational Recurrent Adversarial Deep Domain Adaptation** [[paper]](https://openreview.net/pdf?id=rk9eAFcxg)
- [KDD 2020] **Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data** [[paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403228)
- [AAAI 2021] **Time Series Domain Adaptation Via Sparse Associative Structure Alignment** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16846/16653)
- [IJCAI 2021] **Adversarial Spectral Kernel Matching for Unsupervised Time Series Domain Adaptation** [[paper]](http://palm.seu.edu.cn/hxue/papers/Adversarial%20spectral%20kernel%20matching%20for%20unsupervised%20time%20series%20domain%20adaption.pdf)
- [CHIL 2021] **An Empirical Framework for Domain Generalization in Clinical Settings** [[paper]](https://dl.acm.org/doi/pdf/10.1145/3450439.3451878)
- [TNNLS 2022] **Self-Supervised Autoregressive Domain Adaptation for Time Series Data** [[paper]](https://ieeexplore.ieee.org/iel7/5962385/6104215/09804766.pdf?casa_token=RTkDlGkCGEAAAAAA:4Fom6O1eqOMzJziD-1dN1N29GtUmj_BI3OIYAP_sVfrEAMF0bRsbFlVLHjTR_EAmPaYrX5vIC2sj)
- [MM 2022] **Domain Adaptation for Time-series Classification to Mitigate Covariate Shift** [[paper]](https://dl.acm.org/doi/abs/10.1145/3503161.3548167)
- [ICRA 2022] **Causal-based Time Series Domain Generalization for Vehicle Intention Prediction** [[paper]](https://ieeexplore.ieee.org/iel7/9811522/9811357/09812188.pdf?casa_token=e2xDhMqO680AAAAA:RuW4P1FXIX69pNB5LGuerK2kjPB4EMvf8eyg13aW5kWcx8-JhsVpo-aRj5UUFShDcfTQpbi0Mpw6)
- [ICML 2022] **Domain Adaptation for Time Series Forecasting via Attention Sharing** [[paper]](https://proceedings.mlr.press/v162/jin22d/jin22d.pdf)

### Video
- [ICCV 2019] **Temporal Attentive Alignment for Large-scale Video Domain Adaptation** [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Temporal_Attentive_Alignment_for_Large-Scale_Video_Domain_Adaptation_ICCV_2019_paper.pdf)
- [ECCV 2020] **Shuffle and Attend: Video Domain Adaptation** [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570664.pdf)
- [ICCV 2021] **Learning Cross-modal Contrastive Features for Video Domain Adaptation** [[paper]](http://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Learning_Cross-Modal_Contrastive_Features_for_Video_Domain_Adaptation_ICCV_2021_paper.pdf)
- [NeurIPS 2021] **Contrast and Mix: Temporal Contrastive Video Domain Adaptation with Background Mixing** [[paper]](https://proceedings.neurips.cc/paper/2021/file/c47e93742387750baba2e238558fa12d-Paper.pdf)
- [ECCV 2022] **Source-free Video Domain Adaptation by Learning Temporal Consistency for Action Recognition** [[paper]](https://openreview.net/pdf?id=Jd2WAZomD8r)
- [WACV 2022] **Domain Generalization Through Audio-visual Relative Norm Alignment in First Person Action Recognition** [[paper]](https://openaccess.thecvf.com/content/WACV2022/papers/Planamente_Domain_Generalization_Through_Audio-Visual_Relative_Norm_Alignment_in_First_Person_WACV_2022_paper.pdf)

### Speech
- [SPL 2014] **Autoencoder-based Unsupervised Domain Adaptation for Speech Emotion Recognition** [[paper]](https://ieeexplore.ieee.org/iel7/97/4358004/06817520.pdf?casa_token=BtB4htXN5ZEAAAAA:NIbwEWYQN1pY2zL8WuLt5YMOPL4P9TJes_LlrOr3QeQgAG_wpdoSj3HYmon80YFQ81L7dAy2BE77)
- [ICASSP 2015] **Supervised Domain Adaptation for Emotion Recognition from Speech** [[paper]](https://ieeexplore.ieee.org/iel7/7158221/7177909/07178934.pdf?casa_token=f2ORpk1V0HUAAAAA:NnEYZ7LUMuyGM4NCT6v1ZqE8_ngQfhrhCyl3Ca9JT5W0OpK-5sdtyofX_Hxf0bgSrBeQLSoFi7Mz)
- [NeuroComputing 2017] **An Unsupervised Deep Domain Adaptation Approach for Robust Speech Recognition** [[paper]](https://www.sciencedirect.com/science/article/pii/S0925231217301492?casa_token=UU2tzU41VloAAAAA:F4mnzwMpjSqPZIyeY9Pn-gf1PwbcavrqXorANTc4ROiFmhpnBophRyzKDxvvnS3y3v0O0h-Xnl8g)
- [ASRU 2019] **Domain Adaptation Via Teacher-student Learning for End-to-end Speech Recognition** [[paper]](https://ieeexplore.ieee.org/iel7/8985378/9003727/09003776.pdf?casa_token=0tOiorcsjfwAAAAA:oHUJVPvDvuamvIhbczSXuc-ioli3SsVldPfv36noXDGnMCRVJNEGiY5lJoorYNvP2DrM9J-TsaWE)
- [TAC 2021] **Improving Cross-corpus Speech Emotion Recognition with Adversarial Discriminative Domain Generalization (addog)** [[paper]](https://ieeexplore.ieee.org/iel7/5165369/5520654/08713918.pdf?casa_token=MBUOb9dzmJwAAAAA:YX-po4wbOviRVdYCR412Y6UbI4xgx1ff28dIGlb2qgQEDcTdnkZRQYmOZkOBEDAjphLlBUCUcKoE)
- [SLT 2021] **Domain Generalization with Triplet Network for Cross-corpus Speech Emotion Recognition** [[paper]](https://ieeexplore.ieee.org/iel7/9383468/9383452/09383534.pdf?casa_token=5mLtgBnS4G4AAAAA:GstUzDn2mKQylWq44KNZ7EAaeE-caU8h34jNqukejKThNrVaAgo8ndHgVt3P3HC2uFj51R9dXbfW)
- [ICASSP 2022] **Large-scale Asr Domain Adaptation Using Self-and Semi-supervised Learning** [[paper]](https://ieeexplore.ieee.org/iel7/9745891/9746004/09746719.pdf?casa_token=aKezGIHnTOsAAAAA:WqAJ11D0UFsPX-hJrZIKpOXUB-rqExnBfT9zeuyZuc8LUx-X4-SRnf3U7FV2yP6EuxLhgvsFp7z0)


### Tabular Data
- [NeurIPS 2022] **Distribution-Informed Neural Networks for Domain Adaptation Regression** [[paper]](https://openreview.net/pdf?id=8hoDLRLtl9h)
- [NeurIPS 2022] **C-Mixup: Improving Generalization in Regression** [[paper]](https://arxiv.org/pdf/2210.05775)

### Others (RecSys)
- [KDD 2021] **Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System** [[paper]](https://arxiv.org/pdf/2010.15363)
- [WWW 2021] **Disentangling User Interest and Conformity for Recommendation with Causal Embedding** [[paper]](https://hexiangnan.github.io/papers/www21-dice.pdf)
- [SIGIR 2021] **Causal Intervention for Leveraging Popularity Bias in Recommendation** [[paper]](https://arxiv.org/pdf/2105.06067)
- [SIGIR 2021] **Clicks can be Cheating: Counterfactual Recommendation for Mitigating Clickbait Issue** [[paper]](https://arxiv.org/pdf/2009.09945)
- [SIGIR 2021] **AutoDebias: Learning to Debias for Recommendation** [[paper]](https://hexiangnan.github.io/papers/sigir21-AutoDebias.pdf)
- [WWW 2022] **Cross Pairwise Ranking for Unbiased Item Recommendation** [[paper]](https://hexiangnan.github.io/papers/www22-cpr.pdf)
- [WWW 2022] **Causal Representation Learning for Out-of-Distribution Recommendation** [[paper]](https://hexiangnan.github.io/papers/www22-ood-rec.pdf)
- [SIGIR 2022] **Interpolative Distillation for Unifying Biased and Debiased Recommendation** [[paper]](https://hexiangnan.github.io/papers/sigir22-InterD.pdf)
- [TOIS 2022] **Addressing Confounding Feature Issue for Causal Recommendation** [[paper]](https://arxiv.org/pdf/2205.06532.pdf)
- [TKDE 2023] **Popularity Bias Is Not Always Evil: Disentangling Benign and Harmful Bias for Recommendation** [[paper]](https://hexiangnan.github.io/papers/tkde23-TIDE.pdf)

...

## Decentralized (Federated)
Besides generalization of centralized learning, transferability of decentralized setting (Federated Learning) has also received attention.
- [arxiv 2018] **Federated Learning with Non-IID Data** [[paper]](https://arxiv.org/pdf/1806.00582)
- [ICML 2019] **Agnostic Federated Learning** [[paper]](http://proceedings.mlr.press/v97/mohri19a/mohri19a.pdf)
- [MLSys 2020] **Federated Optimization in Heterogeneous Networks** [[paper]](https://arxiv.org/pdf/1812.06127)
- [ICML 2020] **SCAFFOLD: Stochastic Controlled Averaging for Federated Learning** [[paper]](https://arxiv.org/pdf/1910.06378)
- [NeurIPS 2020] **Robust Federated Learning: the Case of Affine Distribution Shifts** [[paper]](https://proceedings.neurips.cc/paper/2020/file/f5e536083a438cec5b64a4954abc17f1-Paper.pdf)
- [ICLR 2021] **Fedbn: Federated Learning on Non-iid Features Via Local Batch Normalization** [[paper](https://arxiv.org/pdf/2102.07623)
- [SDM 2021] **Fairness-aware Agnostic Federated Learning** [[paper]](https://epubs.siam.org/doi/pdf/10.1137/1.9781611976700.21)
- [TPDS 2022] **Flexible Clustered Federated Learning for Client-level Data Distribution Shift** [[paper]](https://ieeexplore.ieee.org/iel7/71/4359390/09647969.pdf?casa_token=ppLa63KqI0kAAAAA:9tkLoz7O69FS2uFwzU80K8PHdMxEk7CPTBg5YRHs6JigACFKJJuNRvupZ1U8-a3GrBom2KMBEQ)
- [CVPR 2022] **Rethinking Architecture Design for Tackling Data Heterogeneity in Federated Learning** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Qu_Rethinking_Architecture_Design_for_Tackling_Data_Heterogeneity_in_Federated_Learning_CVPR_2022_paper.pdf)
- [CVPR 2022] **Feddc: Federated Learning with Non-iid Data Via Local Drift Decoupling and Correction** [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Gao_FedDC_Federated_Learning_With_Non-IID_Data_via_Local_Drift_Decoupling_CVPR_2022_paper.pdf)

## Detection
Besides generalization, other perspectives including detection, fairness robustness, etc. are also studied.

Out-of-distribution detection, outlier detection and anomaly detection.
- [ICDM 2018] **Adversarially Learned Anomaly Detection** [[paper]](https://ieeexplore.ieee.org/iel7/8591042/8594809/08594897.pdf?casa_token=BpGIhSrJhSsAAAAA:HsmEM6bEK8vWAobZa-XzGhRCzwQV5-y4SsaDQS5er2TxRewJXhS8FltoEJOiujoT3mM4O5DGnQ)
- [arxiv 2018] **Learning Confidence for Out-of-distribution Detection in Neural Networks** [[paper]](https://arxiv.org/pdf/1802.04865)
- [NeurIPS 2019] **Likelihood Ratios for Out-of-distribution Detection** [[paper]](https://proceedings.neurips.cc/paper/2019/file/1e79596878b2320cac26dd792a6c51c9-Paper.pdf)
- [IJCNN 2019] **XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning** [[paper]](https://arxiv.org/pdf/1912.00290)
- [ICDM 2020] **Copod: Copula-based Outlier Detection** [[paper]](https://ieeexplore.ieee.org/iel7/9338245/9338248/09338429.pdf?casa_token=KJ7C5SHVJnAAAAAA:HSpMqKX4ovkFnSM7E286Ri4XvgkPo_knawVVDRzDuGLLYsoBAsLkaKlZH_AAjlSzzLHO9Q_A8w)
- [NeurIPS 2020] **Energy-based Out-of-distribution Detection** [[paper]](https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf)
- [CVPR 2020] **Learning Memory-guided Normality for Anomaly Detection** [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf)
- [NeurIPS 2021] **Exploring the Limits of Out-of-distribution Detection** [[paper]](https://proceedings.neurips.cc/paper/2021/file/3941c4358616274ac2436eacf67fae05-Paper.pdf)
- [NeurIPS 2021] **Automatic Unsupervised Outlier Model Selection** [[paper]](https://proceedings.neurips.cc/paper/2021/file/23c894276a2c5a16470e6a31f4618d73-Paper.pdf)
- [ICCV 2021] **Divide-and-assemble: Learning Block-wise Memory for Unsupervised Anomaly Detection** [[paper]](http://openaccess.thecvf.com/content/ICCV2021/papers/Hou_Divide-and-Assemble_Learning_Block-Wise_Memory_for_Unsupervised_Anomaly_Detection_ICCV_2021_paper.pdf)
- [CVPR 2021] **Anomaly Detection in Video Via Self-supervised and Multi-task Learning** [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Georgescu_Anomaly_Detection_in_Video_via_Self-Supervised_and_Multi-Task_Learning_CVPR_2021_paper.pdf)
- [AAAI 2022] **On the Impact of Spurious Correlation for Out-of-distribution Detection** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/21244/20993)
- [AAAI 2022] **Lunar: Unifying Local Outlier Detection Methods Via Graph Neural Networks** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20629/20388)
- [ICML 2022] **Poem: Out-of-distribution Detection with Posterior Sampling** [[paper]](https://proceedings.mlr.press/v162/ming22a/ming22a.pdf)
- [ECCV 2022] **Dice: Leveraging Sparsification for Out-of-distribution Detection** [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840680.pdf)

## Fairness
- [FAccT 2021] **Fairness Violations and Mitigation Under Covariate Shift** [[paper]](https://dl.acm.org/doi/pdf/10.1145/3442188.3445865)
- [AAAI 2021] **Robust Fairness Under Covariate Shift** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17135/16942)
- [NeurIPS 2022] **Diagnosing Failures of Fairness Transfer Across Distribution Shift in Real-world Medical Settings** [[paper]](https://openreview.net/pdf?id=K-A4tDJ6HHf)
- [NeurIPS 2022] **Fairness Transferability Subject to Bounded Distribution Shift** [[paper]](https://arxiv.org/pdf/2206.00129)
- [NeurIPS 2022] **Transferring Fairness under Distribution Shifts via Fair Consistency Regularization** [[paper]](https://arxiv.org/pdf/2206.12796)
- [arxiv 2022] **How Robust is Your Fairness? Evaluating and Sustaining Fairness under Unseen Distribution Shifts** [[paper]](https://arxiv.org/pdf/2207.01168)

## Robustness
- [ICLR 2019] **On the Sensitivity of Adversarial Robustness to Input Data Distributions** [[paper]](https://arxiv.org/pdf/1902.08336)
- [CVPR 2021] **Adversarial Robustness under Long-Tailed Distribution** [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Adversarial_Robustness_Under_Long-Tailed_Distribution_CVPR_2021_paper.pdf)
- [arxiv 2022] **BOBA: Byzantine-Robust Federated Learning with Label Skewness** [[paper]](https://arxiv.org/pdf/2208.12932)
- [arxiv 2022] **Generalizability of Adversarial Robustness Under Distribution Shifts** [[paper]](https://arxiv.org/pdf/2209.15042)


## Learning Strategy
- [ICLR 2020] **Learning to Balance: Bayesian Meta-learning for Imbalanced and Out-of-distribution Tasks** [[paper]](https://arxiv.org/pdf/1902.08336)
- [NeurIPS 2020] **OOD-MAML: Meta-learning for few-shot out-of-distribution detection and classification** [[paper]](https://proceedings.neurips.cc/paper/2020/file/28e209b61a52482a0ae1cb9f5959c792-Paper.pdf)
- [NeurIPS 2020] **Task-robust Model-agnostic Meta-learning** [[paper]](https://proceedings.neurips.cc/paper/2020/file/da8ce53cf0240070ce6c69c48cd588ee-Paper.pdf)
- [TIP 2021] **Domain Adaptive Ensemble Learning** [[paper]](https://ieeexplore.ieee.org/iel7/83/4358840/09540778.pdf?casa_token=tHapCRZf2aEAAAAA:q2BwltKkK2kZT37g1yjmHTG70IYrcPay4RMq0m7jMWrqFntuMuRnSxYGYYytxXyLIIc9NjiyuQ)
- [NeurIPS 2021] **Two Sides of Meta-Learning Evaluation: In vs. Out of Distribution** [[paper]](https://proceedings.neurips.cc/paper/2021/file/1e932f24dc0aa4e7a6ac2beec387416d-Paper.pdf)
- [ICLR 2022] **Deep Ensembling with No Overhead for either Training or Testing: The All-Round B- lessings of Dynamic Sparsity** [[paper]](https://arxiv.org/pdf/2106.14568)
- [NeurIPS 2022] **Improving Multi-Task Generalization via Regularizing Spurious Correlation** [[paper]](https://arxiv.org/pdf/2205.09797.pdf)
...
