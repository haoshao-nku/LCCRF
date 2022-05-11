# Leaning Compact and Representative Features for Cross-Modality Person Re-Identification
Pytorch code for "Leaning Compact and Representative Features for Cross-Modality Person Re-Identification"(World Wide Web,CCF-B).

## [Highlights]
1.We devise an efficient Enumerate Angular Triplet (EAT) loss, which can better help to obtain an angularly separable common feature space via
explicitly restraining the internal angles between different embedding features, contributing to the improvement of the performance.
2.Motivated by the knowledge distillation, a novel Cross-Modality Knowledge Distillation (CMKD) loss is proposed to reduce the modality discrepancy in the modality-specific feature extraction stage, contributing to the
effectiveness of the cross-modality person Re-ID task.
3.Our network achieves prominent results on both SYSU-MM01 and RegDB datasets without any other data augment skills. It achieves a Mean Average Precision (mAP) of 43.09% and 79.92% on SYSU-MM01 and RegDB datasets, respectively.

## [Prerequisite]
Python>=3.6
Pytorch>=1.0.0
Opencv>=3.1.0
tensorboard-pytorch
## [Experiments]
Training:
'''
python main.py -a train
'''
Testing: 
'''
python main.py -a test -m checkpoint_name -s test_setting
'''
The test settings of SYSU-MM01 include: "all_multi" (all search mode, multi-shot), "all_single" (all search mode, single-shot), "indoor_multi" (indoor search mode, multi-shot), "indoor_single" (indoor search mode, single-shot).

## [Cite]
If you find our paper/codes useful, please kindly consider citing the paper:
@article{gao2022leaning,
  title={Leaning compact and representative features for cross-modality person re-identification},
  author={Gao, Guangwei and Shao, Hao and Wu, Fei and Yang, Meng and Yu, Yi},
  journal={World Wide Web},
  pages={1--18},
  year={2022},
  publisher={Springer}
}
