# awesome-distribution-shift ![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) ![Awesome](https://awesome.re/badge.svg)

A curated list of papers and resources about the distribution shift in machine learning. I categorize them based on their topic and content. I will try to make this list updated.

Here is an example of distribution shift in images across domains from [DomainBed](https://github.com/facebookresearch/DomainBed).

![avatar](https://github.com/weitianxin/awesome-distribution-shift/blob/main/example.png)

I categorize the papers on distribution shift as follows. If you found any error or any missed paper, please don't hesitate to add.

Continuously updated


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
![avatar](https://github.com/weitianxin/awesome-distribution-shift/blob/main/dis%20shift.png)
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
