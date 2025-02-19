# Low-Rank Tensor Function with Schatten-$p$ Quasi-Norm for Implicit Neural Representation

## Installation

1. Download source code and dataset:
    * `git clone https://github.com/CZY-Code/deep-tensor-rank.git`
    * Download the datasets
        - [ShapeNet](https://shapenet.org/)
        - [Color Image](https://sipi.usc.edu/database/database.php)
        - [CAVE](https://www.cs.columbia.edu/CAVE/databases/multispectral/)
        - [WDC Mall and PaviaU](https://rslab.ut.ac.ir/data)
        - [Videos](http://trace.eas.asu.edu/yuv/)
        - [Bunny](https://graphics.stanford.edu/data/3Dscanrep/)
   
2.  Pip install dependencies:
    * OS: Ubuntu 20.04.6
    * nvidia :
        - cuda: 12.1
        - cudnn: 8.5.0
    * python == 3.9.18
    * pytorch >= 2.1.0
    * Python packages: `pip install -r requirements.txt`

## Dataset Preparation
Unzip and move dataset into ROOT/dataset or ROOT/data

### Directory structure of dataset          
        ├── data                
        │   ├── misc              
        │   ├── MSIs
        |   ├── Videos
        |   dc.tif
        |   PaviaU.mat          
        ├── dataset
        │   ├── bunny         
        │   ├── shapeNet

## Run and test
* Inpainting: `./Inpainting.sh`
* Denoising: `./Denoising.sh`
* Upsampling `./Upsampling.sh`
    
## Acknowledgement
This implementation is based on / inspired by:
* [reproducible-tensor-completion-state-of-the-art](https://github.com/zhaoxile/reproducible-tensor-completion-state-of-the-art)
* [M $^2$ DMT](https://github.com/jicongfan/Multi-Mode-Deep-Matrix-and-Tensor-Factorization)
* [HLRTF](https://github.com/YisiLuo/HLRTF)
* [Continuous-Tensor-Toolbox](https://github.com/YisiLuo/Continuous-Tensor-Toolbox)
* [DeepTensor](https://github.com/vishwa91/DeepTensor)
* [SAPCU](https://github.com/xnowbzhao/sapcu)
* [NeuralTPS](https://github.com/chenchao15/NeuralTPS)
* [NeuralPoints](https://github.com/WanquanF/NeuralPoints)
* [Grad-PU](https://github.com/yunhe20/Grad-PU)