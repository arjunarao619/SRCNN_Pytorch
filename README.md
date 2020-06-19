# SRCNN Pytorch Implementation

### Implementation of [https://arxiv.org/abs/1501.00092](https://arxiv.org/abs/1501.00092) - Image Super-Resolution Using Deep Convolutional Networks ([Original Caffe Model](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)) on Holopix50k Stereo Dataset

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

Instructions to download Holopix50k can be found on [Holopix50k repository](https://github.com/leiainc/holopix50k) ([Paper](https://arxiv.org/abs/2003.11172)). Training on 5000 images of Holopix50k with 120 X 78 patch and stride yields ~500,000 patches of size `(120 X 120)`. Random crop patches can be decreased by altering size in `preprocess.py`. Output dataset is stored at `output/<dataset.h5>`

**Note:**  Our crop sizes are significantly higher than original paper's size. Authors train with a crop size of 33 and stride 14, resulting in ~ 24,000 images from the 91-image dataset which has average resolution of `(180X150)`. Moreover, we apply a gaussian kernel of a higher standard deviation (0.55 in paper vs 0.73 ours) Hence, our input images have a much lower quality compared to the original implementation and make our training more challenging.

### Model
SRCNN 9-5-5 was chosen due to nature of large resolution. 
Chosen Network Settings: <a href="https://www.codecogs.com/eqnedit.php?latex={&space;f&space;}_{&space;1&space;}=9,{&space;f&space;}_{&space;2&space;}=5,{&space;f&space;}_{&space;3&space;}=5,{&space;n&space;}_{&space;1&space;}=64,{&space;n&space;}_{&space;2&space;}=32,{&space;n&space;}_{&space;3&space;}=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{&space;f&space;}_{&space;1&space;}=9,{&space;f&space;}_{&space;2&space;}=5,{&space;f&space;}_{&space;3&space;}=5,{&space;n&space;}_{&space;1&space;}=64,{&space;n&space;}_{&space;2&space;}=32,{&space;n&space;}_{&space;3&space;}=1" title="{ f }_{ 1 }=9,{ f }_{ 2 }=5,{ f }_{ 3 }=5,{ n }_{ 1 }=64,{ n }_{ 2 }=32,{ n }_{ 3 }=1" /></a>.

### Training

Training was performed under paper conditions for 100 epochs using MSE Loss as training loss. Training was stopped when Validation PSNR flattened at ~80.10. 

SSIM was calculated during eval. Similarity score peaked at 0.99, thus suggesting that SSIM is not reliable for SR tasks.




## Saved Weights

* Saved Weights for training using original MSE Loss can be downloaded [here](https://drive.google.com/file/d/1JUfM9vzzaSlyVS3_4xACBEwZv1FFSuhC/view?usp=sharing)
* Saved weights for training using our weighted loss can be downloaded [here](https://drive.google.com/file/d/1Jq-fWU-htYqMfIs6jTl1Rk8oFzUrWgAn/view?usp=sharing)

### Results 

<table>
    <tr>
        <td><center>Synthesized Low Res Image</center></td>
        <td><center>BICUBIC (X3)</center></td>
        <td><center>SRCNN (X3) , PSNR ~= 87.30</center></td>
        <td><center>Original</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./results/coffee_srcnn_0.png""></center>
    	</td>
    	<td>
    		<center><img src="./results/coffee_bicubic_30.png"></center>
    	</td>
    	<td>
    		<center><img src="./results/coffee_srcnn_30.png"></center>
    	</td>
        <td>
    		<center><img src="./results/coffee.png"></center>
    	</td>
    </tr>
    <tr>
        <td><center>Synthesized Low Res Image</center></td>
        <td><center>BICUBIC (X3)</center></td>
        <td><center>SRCNN (X3) , PSNR ~= 82.10</center></td>
        <td><center>Original</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./results/flower_srcnn_0.png""></center>
    	</td>
    	<td>
    		<center><img src="./results/flower_bicubic_30.png"></center>
    	</td>
    	<td>
    		<center><img src="./results/flower_srcnn_30.png"></center>
    	</td>
        <td>
    		<center><img src="./results/flower.png"></center>
    	</td>
    </tr>
    <tr>
        <td><center>Synthesized Low Res Image</center></td>
        <td><center>BICUBIC (X3)</center></td>
        <td><center>SRCNN (X3) , PSNR ~= 84.21</center></td>
        <td><center>Original</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./results/doggie_srcnn_0.png""></center>
    	</td>
    	<td>
    		<center><img src="./results/doggie_bicubic_30.png"></center>
    	</td>
    	<td>
    		<center><img src="./results/doggie_srcnn_30.png"></center>
    	</td>
        <td>
    		<center><img src="./results/doggie.png"></center>
    	</td>
    </tr>
      <tr>
        <td><center>Synthesized Low Res Image</center></td>
        <td><center>BICUBIC (X3)</center></td>
        <td><center>SRCNN (X3) , PSNR ~= 85.53</center></td>
        <td><center>Original</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./results/child_srcnn_0.jpg""></center>
    	</td>
    	<td>
    		<center><img src="./results/child_bicubic_30.jpg"></center>
    	</td>
    	<td>
    		<center><img src="./results/child_srcnn_30.jpg"></center>
    	</td>
        <td>
    		<center><img src="./results/child.jpg"></center>
    	</td>
    </tr>
</table>

### Experiments

We have also trained using a weighted loss function from 3 image reconstruction metrics
* Structural Similarity Index (SSIM)
* Peak Signal-to-Noise Ratio (PSNR)
* Mean Square Error (MSE)

We have weighted these according to result priority. Our current weighting is:
<div style="text-align: center">
<a href='https://i.postimg.cc/75PgrGR4/fil.png' target='_blank'><img src='./results/eq.png' border='0'  alt='filters'/></a>
</div>


We have also trained out model using Peak Signal-to-Noise Ratio (PSNR Score)  as a loss function. Below are our results:

* **Weighted Loss:** We are able to reduce the heavy gaussian blurring with our weighted loss at the cost of loosing slight structural information.
* **PSNR Loss:** Similar results as weighted loss.
	### Training with Weighted Loss

	<table>
    <tr>
        <td><center>MSE Trained Result</center></td>
        <td><center>Weighted Trained Result</center></td>
 
    </tr>
    <tr>
    	<td>
    		<center><img src="./results/child_srcnn_30.jpg""></center>
    	</td>
    	<td>
    		<center><img src="./results/child_srcnn_weighted_47.jpg"></center>
    	</td>
 
    </tr>
    <tr>
        <td><center>MSE Trained Result</center></td>
        <td><center>Weighted Trained Result</center></td> 
    </tr>
    <tr>
    	<td>
    		<center><img src="./results/flower_srcnn_30.png""></center>
    	</td>
    	<td>
    		<center><img src="./results/flower_srcnn_weighted_47.png"></center>
    	</td>  
    </tr>
	</table>
	
	### Training on PSNR Only

	We observed very similar results to our weighted loss after using 1-PSNR as a loss function. Please Zoom in to see slight structural changes. 
	<table>
	<tr>
        <td><center>MSE Trained Result</center></td>
        <td><center>PSNR Trained Result</center></td> 
		<td><center>Original</center></td> 
    </tr>
    <tr>
    	<td>
    		<center><img src="./results/idol_srcnn_mse_90.jpg""></center>
    	</td>
    	<td>
    		<center><img src="./results/idol_srcnn_psnr_9.jpg"></center>
    	</td>  
		<td>
    		<center><img src="./results/idol.jpg"></center>
    	</td> 
    </tr>
	</table>

### Visualized Filters (Optional)

We visualize our trained Conv filters of size `(9X9)` (Trained with MSE Loss). We can identify that some filters are edge vs texture detectors. This is reflective of the original implementation's findings.
<div style="text-align: center">
<a href='https://i.postimg.cc/75PgrGR4/fil.png' target='_blank'><img src='https://i.postimg.cc/75PgrGR4/fil.png' border='0'  alt='filters'/></a>
</div>

### TODO
* ~~Save filters during training~~
* Add accelerated SRCNN model with deconvolutions
* ~~Train with SSIM, MSSIM, PSNR opposed to MSE Loss specified in paper~~
* Pretrain Y channel and CB, Cr channels for better results






