## COOL: Contour and Object-Oriented Learning for Indoor Cleanliness Classification

## Abstract
<span style="font-size: 1.5em;">The evaluation of indoor cleanliness is a meaningful task for vision-based household service systems. However, the perception of cleanliness is determined by diverse visual features and multiple criteria, which is subjective to the observer. We find the existing dataset and method fail to truthfully capture the concept of cleanliness because the feature used is not representative to human subjective judgement. Therefore, we create a dataset for indoor cleanliness classification from a group of annotators based on SUN-RGBD, a richly annotated scene understanding benchmark. Based on such analysis, we propose Contour and Object-oriented Learning (COOL) model that integrates pretrained convolutional feature, low-level contour feature, and object arrangement in order to truthfully model the notion of cleanliness. Our design choices are justified in ablation studies, and our model outperforms the previous method in our dataset for cleanliness classification.</span>

<center><img src="./doc/cool-model-arch.png" align="middle" width="850"></center> 

## Authors 

<table style="width:100% bgcolor:#FFFFFF" align="center">
  <tr align="center">
    <th><a href="https://github.com/OolongQian">Sucheng Qian</a></th>
    <th><a href="https://github.com/ApolloLiZhaoyu">Zhaoyu Li</a></th>
    <th>Weibang Jiang</th>
  </tr>
</table>

## Demo 
<span style="font-size: 1.5em;"> An introduction video to this project can be downloaded here [link](https://youtu.be/5RG4RuoPPo8).</span>

## Paper

- Paper in PDF format is availale [here](https://github.com/OolongQian/Cleanliness-Classification/tree/master/doc/paper.pdf).

- Citation in bibtex availale [here](https://github.com/OolongQian/Cleanliness-Classification/tree/master/doc/bibtex_paper.txt).

## Code 
### Prerequisites
- Linux / macOS
- NVIDIA GPU with CUDA CuDNN
- Python 3

### Getting Started

- Clone this repo
```bash
git clone https://github.com/OolongQian/Cleanliness-Classification
cd Cleanliness-Classification
```

- Install requirements 
```bash 
pip install -r requirements.txt
```

- Data preparation (Please wait for Google Drive to complete uploading)
  
  Download project [dataset](https://drive.google.com/file/d/1tVgTA-8oWswWkQZNoIoHUI9qyxB8QfmT/view?usp=sharing) shared via Google Drive. And extract it as ./data folder under repository root. 
  
  *Please refer to the paper for the details of constructed dataset.*

- Model training with testing
```bash
python3 train_cool.py
```

- Run baseline
```bash
python3 train_visual.py
```

- The train and test performance are logged into ./runs folder. 

### Citation
    @article{Qian2019COOL
        author = {Sucheng, Qian and Zhaoyu, Li and Weibang, Jiang",
        title = {Contour and Object-Oriented Learning for Indoor Cleanliness Classification},
        year = {2019},
        howpublished={\url{https://github.com/OolongQian/Cleanliess-Classification}}
    }

### Acknowledgements 
- The images and object-centric annotations of this dataset are created out of SUNRGBD. 
- The image contours are processed using Holistically-Nested Edge Detection [1], implemented in PyTorch by [2]. 


## references
```
[1]  @inproceedings{Xie_ICCV_2015,
         author = {Saining Xie and Zhuowen Tu},
         title = {Holistically-Nested Edge Detection},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2015}
     }
```

```
[2]  @misc{pytorch-hed,
         author = {Simon Niklaus},
         title = {A Reimplementation of {HED} Using {PyTorch}},
         year = {2018},
         howpublished = {\url{https://github.com/sniklaus/pytorch-hed}}
    }
```
