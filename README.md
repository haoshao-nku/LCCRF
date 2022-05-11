# Leaning Compact and Representative Features for Cross-Modality Person Re-Identification
Pytorch code for "Leaning Compact and Representative Features for Cross-Modality Person Re-Identification"(World Wide Web,CCF-B).

## [Highlights]
<br>1.We devise an efficient Enumerate Angular Triplet (EAT) loss, which can better help to obtain an angularly separable common feature space via explicitly restraining the <br>internal angles between different embedding features, contributing to the improvement of the performance.
<br>2.Motivated by the knowledge distillation, a novel Cross-Modality Knowledge Distillation (CMKD) loss is proposed to reduce the modality discrepancy in the modality-<br>specific feature extraction stage, contributing to the effectiveness of the cross-modality person Re-ID task.
<br>3.Our network achieves prominent results on both SYSU-MM01 and RegDB datasets without any other data augment skills. It achieves a Mean Average Precision (mAP) of <br>43.09% and 79.92% on SYSU-MM01 and RegDB datasets, respectively.

## [Prerequisite]
<br>Python>=3.6
<br>Pytorch>=1.0.0
<br>Opencv>=3.1.0
<br>tensorboard-pytorch
## [Experiments]
Training:
<br>python main.py -a train
<br>Testing: 
<br>python main.py -a test -m checkpoint_name -s test_setting
<br>The test settings of SYSU-MM01 include: "all_multi" (all search mode, multi-shot), "all_single" (all search mode, single-shot), "indoor_multi" (indoor search mode, multi-shot), "indoor_single" (indoor search mode, single-shot).

## [Cite]
If you find our paper/codes useful, please kindly consider citing the paper:
<br>@article{gao2022leaning,
<br>  title={Leaning compact and representative features for cross-modality person re-identification},
<br>  author={Gao, Guangwei and Shao, Hao and Wu, Fei and Yang, Meng and Yu, Yi},
<br>  journal={World Wide Web},
<br>  pages={1--18},
<br>  year={2022},
<br>  publisher={Springer}
<br>}
