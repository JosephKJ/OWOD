## Towards Open World Object Detection
#### CVPR 2021

Humans have a natural instinct to identify unknown object instances in their environments. The intrinsic curiosity about these unknown instances aids in learning about them, when the corresponding knowledge is eventually available. This motivates us to propose a novel computer vision problem called: Open World Object Detection, where a model is tasked to: 
1) Identify objects that have not been introduced to it as `unknown', without explicit supervision to do so, and 
2) Incrementally learn these identified unknown categories without forgetting previously learned classes, when the corresponding labels are progressively received. 

We formulate the problem, introduce a strong evaluation protocol and provide a novel solution, which we call ORE: Open World Object Detector, based on contrastive clustering and energy based unknown identification. Our experimental evaluation and ablation studies analyse the efficacy of ORE in achieving Open World objectives. As an interesting by-product, we find that identifying and characterising unknown instances helps to reduce confusion in an incremental object detection setting, where we achieve state-of-the-art performance, with no extra methodological effort. We hope that our work will attract further research into this newly identified, yet crucial research direction. 
   

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

Data split and trained models: [Google Drive](https://drive.google.com/drive/folders/1Sr4_q0_m2f2SefoebB25Ix3N1VIAua0w?usp=sharing)

All config files can be found in: `configs/OWOD`

Sample command on a 4 GPU machine:
```python
python tools/train_net.py --num-gpus 4 --config-file <Change to the appropriate config file> SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005
```



## Acknowledgement

Our code base is build on top of [Detectron 2](https://github.com/facebookresearch/detectron2) library. 


## Citation

If you use our work in your research please cite us:

```BibTeX
@misc{open_joseph,
  author =       {K J Joseph and Salman Khan and Fahad Khan and Vineeth N Balasubramanian},
  booktitle =    {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year =         {2021}
}
```