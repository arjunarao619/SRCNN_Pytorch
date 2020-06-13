# SRCNN Pytorch Implementation

### Implementation of [https://arxiv.org/abs/1501.00092](arXiv) - Image Super-Resolution Using Deep Convolutional Networks ([Original Caffe Model](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)) on Holopix50k Stereo Dataset

### Requirements:
* Pytorch > 1.4.0
* tqdm >= 4.42.1 (train progress bar can be found [here](https://github.com/tqdm/tqdm))
* TensorboardX >= 2.0
* PIL.Image(Pillow) >= 7.0.0
* Scikit-image >= 0.16.2
* h5py >= 2.10.0

### Additions to common Pytorch implementation:
* Added SSIM, PSNR, and MSSIM as training and logging metrics.
* Trained on high resolution `(640 X 360) Train, (1280 X 720) Test)` Holopix50k images. 

### Dataset

Instructions to download Holopix50k can be found on [Holopix50k repository](https://github.com/leiainc/holopix50k) ([Paper](https://arxiv.org/abs/2003.11172)). Training on 1500 images of Holopix50k with 113 X 74 patch and stride yields ~110,000 patches. Random crop patches can be decreased by altering size in `preprocess.py`. Output dataset is stored at `output/<dataset.h5>`

### Todo
* Save filters during training
* Add accelerated SRCNN model with deconvolutions
* Train with SSIM, MSSIM, PSNR opposed to MSE Loss specified in paper
* Pretrain Y channel and CB, Cr channels for better results





