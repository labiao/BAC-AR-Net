## Weakly Supervised Building Extraction from High-Resolution Remote Sensing Images based on Building-Aware Clustering and Activation Refinement Network(TGRS)

<!-- ![](https://github.com/labiao/BAC-AR-Net/blob/main/figure/BAC-AR-Net.jpg?raw=true){: .center} -->

<p align="center">
<img src="https://github.com/labiao/BAC-AR-Net/blob/main/figure/BAC-AR-Net.jpg?raw=true">
</p>

__Official pytorch implementation of "Weakly Supervised Building Extraction from High-Resolution Remote Sensing Images based on Building-Aware Clustering and Activation Refinement Network"__


## Updates

26 Aug, 2024: Initial upload


## Requirement 

- This code is tested on Ubuntu 20.04, with Python 3.6, PyTorch 1.7.1, and CUDA 11.3.

### Dataset & pretrained checkpoint

- Prepare dataset, image-level labels, and pretrained checkpoints
  - Example directory hierarchy
  ```
  BAC-AR-Net
  |--- Dataset
  |    |--- RRDSD
  |    |        |---JPEGImages
  |    |        |---SegmentationClassAug
  |--- weights
  |    |--- RepVGG-B1g2-train.pth
  |--- sess_rrdsd
  |    |--- repvgg_cam_delpoy.pt
  |--- result_rrdsd
  |    |--- cam
  |    |--- amn_cam
  |    | ...
  | ...
  ```


## Execution

### Pseudo-mask generation

- Execute the bash file.
    ```bash
    # Please see these files for the detail of execution.
    bash script/generate_pseudo_mask.sh
    ```

### Segmentation network
Fort the segmentation network, we experimented with DeepLab-V3+ based on [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9).


## Acknowledgement
This code is highly borrowed from [IRN](https://github.com/jiwoon-ahn/irn), [AMN](https://github.com/gaviotas/AMN.git), [LPCAM](https://github.com/zhaozhengChen/LPCAM.git). Thanks to Jiwoon, Ahn, Minhyun Lee, [Zhaozheng Chen](https://github.com/zhaozhengChen).

The codes for our previous work, including [ACGC](https://github.com/labiao/ACGC.git) and [MFR-PGC-Net](https://github.com/labiao/MFR-PGC-Net.git), are also avaliable. Detailed information can be found in the respective publication papers.


## Citation
If you find this work useful for your research, please cite our paper:
```
@ARTICLE{10623252,
  author={Zheng, Daoyuan and Wang, Shaohua and Feng, Haixia and Wang, Shunli and Ai, Mingyao and Zhao, Pengcheng and Li, Jiayuan and Hu, Qingwu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Weakly Supervised Building Extraction From High-Resolution Remote Sensing Images Based on Building-Aware Clustering and Activation Refinement Network}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  keywords={Buildings;Feature extraction;Semantics;Accuracy;Annotations;Semantic segmentation;Optimization;Building extraction;high-resolution (HR) remote sensing (RS) images;image-level labels;weakly supervised semantic segmentation},
  doi={10.1109/TGRS.2024.3438248}}
```
