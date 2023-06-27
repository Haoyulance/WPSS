<p align=""left>
<img src="https://img.shields.io/badge/release--date-06%2F2023-green.svg">
</p>

# WPSS
The project repository for *Weakly supervised perivascular spaces segmentation with salient guidance of Frangi filter* (WPSS). WPSS is a weakly supervised convolutional neural network model which focuses on segmenting perivascular spaces (PVS) in the white matter regions. 

## Acknowledgement

[**Weakly upervised Perivascular Spaces Segmentation with salient guidance of Frangi filter**<br>](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29593)
*Haoyu Lan, Kirsten M. Lynch, Rachel Custer, Nien‐Chu Shih, Patrick Sherlock, Arthur W. Toga, Farshid Sepehrband, Jeiran Choupan*

For data inquiries, please contact me by the email in the publication. 

<p align="center">
  <img src="https://github.com/Haoyulance/WPSS/blob/main/brain.gif" width="800" height="800" />
</p>

## Data
Both *training* and *inference* datasets should be organized as the following structure:

```
dataset
│
└───subject 1
|   |    epc.nii.gz
|   |    mask.nii.gz(optional)
|   |    target.nii.gz(not required for the inference dataset)
└───subject 2
|   |    epc.nii.gz
|   |    mask.nii.gz
|   |    target.nii.gz
└───subject 3
|   |    epc.nii.gz
|   |    mask.nii.gz
|   |    target.nii.gz
│   ...
```
[Human Connectome Project (HCP)](https://cran.r-project.org/web/packages/neurohcp/vignettes/hcp.html) dataset was used for the WPSS training and evaluation. 

**The Enhanced PVS Contrast (EPC) modality and training target generations are upon request**. Please refer to the [original publication](https://link.springer.com/content/pdf/10.1038/s41598-019-48910-x.pdf) for additional information of EPC modality. 

## Requirements

python 3 is required and `python 3.6.4` was used in the study.

Requirements can be found at [requirements.txt](https://github.com/Haoyulance/WPSS/blob/main/requirements.txt).

Please use ```pip install requirements.txt``` to install the requirements



## How to run the code
Use the following command for the model training and inference. **Pretrained weights with quality controlled labels for the model inference using either the EPC modality or the T1w modality are upon request**. 

### Training:
Training script is at  **./src/training**

Use the following command to run the training script:

```python training.py --trainig_size= --gpu= --num_iter= --patch_size= --dataset= --modality= --logdir= --output= --val_portion=```

|configurations|meaning|default|
|---|---|---|
|--training_size|the number of training data to use|None|
|--gpu|gpu ID for training|None|
|--val_portion|percentage of the training dataset used for the validation|0.25|
|--num_iter|the number of iteration|3000|
|--patch_size|input image size (same for three dimensions)|16|
|--dataset|path to the training dataset|None|
|--modality|names of the input image and the target image|epc_frangi|
|--logdir|path to the saved tensorboard log|None|
|--output|path to saved images during training|None|
|--lr|learning rate|0.001|
|--mask|if using the mask for the evaluation on validation data (mask should be save as "mask.nii.gz")|False|


### Inference:

Inference script is at  **./src/predict**. The corresponding PVS segmentation results will be saved under the same directory as the inference dataset. 

Use the following command to run the inference script:

```python predict.py --gpu= --weights= --dataset= --modality=```

|configurations|meaning|default|
|---|---|---|
|--gpu|gpu ID for inference|0|
|--weights|path of trained model weights to load|None|
|--patch_size|patch size for three dimensions|16|
|--dataset|path to the inference dataset|None|
|--modality|name of the modality used as the input (could be either "epc" or "t1w")|epc|



## License

If you use this code for your research, please be familiar with the [LICENSE](./LICENSE) and cite our paper.

## Citation 
```
@article{lan2023weakly,
  title={Weakly supervised perivascular spaces segmentation with salient guidance of Frangi filter},
  author={Lan, Haoyu and Lynch, Kirsten M and Custer, Rachel and Shih, Nien-Chu and Sherlock, Patrick and Toga, Arthur W and Sepehrband, Farshid and Choupan, Jeiran},
  journal={Magnetic Resonance in Medicine},
  volume={89},
  number={6},
  pages={2419--2431},
  year={2023},
  publisher={Wiley Online Library}
}
```


