## Automatic Image Captioning

A neural network architecture which automatically generates captions from images.
The architecture and hyperparameters are inspired the work of Vinyals et al. [[1]][show_and_tell] 
and Xu et al. [[2]][show_attend_and_tell].

This is my submission for the image captioning project in the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).

My nanodegree certificate: [https://confirm.udacity.com/SYAMKDHY](https://confirm.udacity.com/SYAMKDHY) 

### Requirements

The model was developed in cloud-hosted JupyterLab environment, with some custom packages from Udacity available. The `requirements.txt` file can 
help you get started reproducing the results. The trained model weights are also included in the `models/` folder.
The weights are stored using [Git LFS](https://git-lfs.github.com/), which needs to be installed before checking out the repository. 

### Network Architecture

![architecture](resources/task_overview.png)
<figcaption align = "center"><b>Image Captioning Model</b></figcaption>

The model is based on a decoder-endcoder architecture. The encoder is a ResNet-50 model trained on the ImageNet dataset. Its final
layer is connected to an embedding layer, whose output serves as the initial input to the LSTM-based RNN decoder.

For the full model architecture please refer to `model.py`.

### Training

I have kept the pre-trained ResNet weights frozen during training. The embedding layer an the decoder were trained from scratch using the [COCO dataset](https://cocodataset.org/#home).

For details on hyperparameter choices, please refer to `2_Training.html`.

### Inference

Inference is implemented using **Sampling**, where at each step of the RNN, the word with the highest softmax probability is selected as output,
and is used as input for the next step after passing it through an embedding layer. Refer to `3_Inference.html` for example outputs 
on the test dataset.

## References

1. [Oriol Vinyals and Alexander Toshev and Samy Bengio and Dumitru Erhan. &nbsp; Show and Tell: A Neural Image Caption Generator. In *arXiv:1411.4555*, 2015.][show_and_tell]
2. [K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R. Zemel and Y. Bengio. &nbsp; Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. In *arXiv:1502.03044*, 2016.][show_attend_and_tell]


[show_and_tell]: https://arxiv.org/pdf/1411.4555.pdf
[show_attend_and_tell]: https://arxiv.org/pdf/1502.03044.pdf