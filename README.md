## Towards Open World Object Detection [[arXiv](https://arxiv.org/abs/2103.02603) | [video](https://www.youtube.com/watch?v=aB2ZFAR-OZg) | [poster](https://github.com/JosephKJ/OWOD/blob/master/ORE_poster.pdf)]
### Presented at CVPR 2021 as an ORAL paper


<p align="center" width="100%">
<img src="https://josephkj.in/assets/img/owod/the_problem.png" width="600"/>
</p>


<p align="center" width="80%">
The figure shows how our newly formulated Open World Object Detection setting relates to exsiting settings.
</p>


#### Abstract

Humans have a natural instinct to identify unknown object instances in their environments. The intrinsic curiosity about these unknown instances aids in learning about them, when the corresponding knowledge is eventually available. This motivates us to propose a novel computer vision problem called: Open World Object Detection, where a model is tasked to: 
1) Identify objects that have not been introduced to it as `unknown', without explicit supervision to do so, and 
2) Incrementally learn these identified unknown categories without forgetting previously learned classes, when the corresponding labels are progressively received. 

We formulate the problem, introduce a strong evaluation protocol and provide a novel solution, which we call ORE: Open World Object Detector, based on contrastive clustering and energy based unknown identification. Our experimental evaluation and ablation studies analyse the efficacy of ORE in achieving Open World objectives. As an interesting by-product, we find that identifying and characterising unknown instances helps to reduce confusion in an incremental object detection setting, where we achieve state-of-the-art performance, with no extra methodological effort. We hope that our work will attract further research into this newly identified, yet crucial research direction. 


#### A sample qualitative result

<p align="center" width="100%">
<img src="https://josephkj.in/assets/img/owod/example.png" width="800" />
</p>

<p align="center" width="80%">
The sub-figure (a) is the result produced by our method after learning a few set of classes which doesnot include classes like <strong>apple</strong> and <strong>orange</strong>. We are able to identify them and correctly labels them as <strong>unknown</strong>. 
After some time, when the model is eventually taught to detect <strong>apple</strong> and <strong>orange</strong>, these instances are labelled correctly as seen in sub-figure (b); without forgetting how to detect <strong>person</strong>. 
An unidentified class instance still remains, and is successfully detected as an <strong>unknown</strong>.
</p>

## Installation

See [INSTALL.md](INSTALL.md).

Dataset setup: Follow [these](https://github.com/JosephKJ/OWOD/issues/59#issuecomment-897747744) instructions.

## Quick Start

Some bookkeeping needs to be done for the code, like removing the local paths and so on. We will update these shortly. 

Data split and trained models: [[Google Drive Link 1]](https://drive.google.com/drive/folders/1Sr4_q0_m2f2SefoebB25Ix3N1VIAua0w?usp=sharing) [[Google Drive Link 2]](https://drive.google.com/drive/folders/11bJRdZqdtzIxBDkxrx2Jc3AhirqkO0YV?usp=sharing)

All config files can be found in: `configs/OWOD`

Sample command on a 4 GPU machine:
```python
python tools/train_net.py --num-gpus 4 --config-file <Change to the appropriate config file> SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005
```

Kindly run `replicate.sh` to replicate results from the models shared on the Google Drive. 

Kindly check `run.sh` file for a task workflow.

## Incremental Object Detection
If you are interested in Incremental Object Detection, you may have to consider [this](https://github.com/JosephKJ/iOD) recent work. We have released the exact training configurations, trained models and logs for all three incremental settings: https://github.com/JosephKJ/iOD 

## Acknowledgement

Our code base is build on top of [Detectron 2](https://github.com/facebookresearch/detectron2) library. 


## Citation

If you use our work in your research please cite us:

```BibTeX
@inproceedings{joseph2021open,
  title={Towards Open World Object Detection},
  author={K J Joseph and Salman Khan and Fahad Shahbaz Khan and Vineeth N Balasubramanian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021)},
  eprint={2103.02603},
  archivePrefix={arXiv},
  year={2021}
}
```
