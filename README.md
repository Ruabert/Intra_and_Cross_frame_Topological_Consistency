# Intra_and_Cross_frame_Topological_Consistency
The official code of [ICASSP 2025] An Intra- and Cross-frame Topological Consistency Scheme for Semi-supervised Atherosclerotic Coronary Plaque Segmentation.

link: https://ieeexplore.ieee.org/document/10890181

## Introduction
Enhancing the precision of segmenting coronary atherosclerotic plaques from CT Angiography (CTA) images is pivotal for advanced Coronary Atherosclerosis Analysis (CAA), which distinctively relies on the analysis of vessel cross-section
images reconstructed via Curved Planar Reformation. This task presents significant challenges due to the indistinct boundaries and structures of plaques and blood vessels, leading to the inadequate
performance of current deep learning models, compounded by the inherent difficulty in annotating such complex data. To address these issues, we propose a novel dual-consistency semisupervised framework that integrates Intra-frame Topological Consistency (ITC) and Cross-frame Topological Consistency (CTC) to leverage labeled and unlabeled data. ITC employs a dual-task network for simultaneous segmentation mask and Skeleton-aware Distance Transform (SDT) prediction, achieving similar prediction of topology structure through consistency constraint without additional annotations. Meanwhile, CTC utilizes an unsupervised estimator for analyzing pixel flow between skeletons and boundaries of adjacent frames, ensuring spatial continuity. Experiments on two CTA datasets show that our method surpasses existing semi-supervised methods and approaches the performance of supervised methods on CAA. In addition, our method also performs better than other methods on the ACDC dataset, demonstrating its generalization.
## Usage
### Pre-processing
Curved Planar Reformation (CPR) tools: https://github.com/Ruabert/CADSegmTools
### Train

### Inference

## Citation
If you use this toolbox or benchmark in your research, please cite this project.
"""
@INPROCEEDINGS{10890181,
  author={Zhang, Ziheng and Li, Zihan and Shan, Dandan and Qiu, Yuehui and Hong, Qingqi and Wu, Qingqiang},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={An Intra- and Cross-frame Topological Consistency Scheme for Semi-supervised Atherosclerotic Coronary Plaque Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Image segmentation;Network topology;Atherosclerosis;Transforms;Signal processing;Skeleton;Topology;Reliability;Speech processing;Image reconstruction;Semi-supervised segmentation;Atherosclerosis analysis;Topological consistency},
  doi={10.1109/ICASSP49660.2025.10890181}}
"""
