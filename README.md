这是一个图像分类相关的代码集合，目标是把自己感兴趣的方法整合到一个统一的框架里，方便以后复现别人的方法以及验证自己的想法。本工程包括以下几个部分：

#### 骨干网络

- [x] ResNet: He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
  经典残差网络
- [ ] VGGNet: Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.
- [ ] MobileNet: Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications[J]. arXiv preprint arXiv:1704.04861, 2017.

#### 注意力机制

- [x] SENet: Hu, J., et al. (2019). "Squeeze-and-Excitation Networks." arXiv pre-print server.
  通道注意力机制中的经典方法
- [x] CBAM: Woo, S., et al. (2018). "CBAM: Convolutional Block Attention Module." arXiv pre-print server.
  通道和空间注意力机制
- [x] Non-local: Wang, X., et al. (2018). "Non-local Neural Networks." arXiv pre-print server.
  自注意力机制的经典方法
- [ ] Coordinate Attention: Hou, Q., et al. (2021). "Coordinate Attention for Efficient Mobile Network Design." arXiv pre-print server.

#### 损失函数

- [x] softmax cross-entropy loss
- [ ] arcface loss: Deng J, Guo J, Xue N, et al. Arcface: Additive angular margin loss for deep face recognition[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 4690-4699.
- [ ] center loss: Wen Y, Zhang K, Li Z, et al. A discriminative feature learning approach for deep face recognition[C]//European conference on computer vision. 2016: 499-515

#### 数据增广

- [ ] cutmix: Yun, S., et al. (2019). "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features." arXiv pre-print server.
- [ ] mixup: Zhang H, Cisse M, Dauphin Y N, et al. mixup: Beyond empirical risk minimization[J]. arXiv preprint arXiv:1710.09412, 2017.
- [ ] AutoAugment: Cubuk E D, Zoph B, Mane D, et al. Autoaugment: Learning augmentation policies from data[EB/OL]. [2020-01-01]. http://arxiv.org/abs/1805.09501.

#### 长尾分布下的分类

- [ ] Zhou B, Cui Q, Wei X S, et al. Bbn: Bilateral-branch network with cumulative learning for long-tailed visual recognition[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 9719-9728.
- [ ] Kang B, Xie S, Rohrbach M, et al. Decoupling representation and classifier for long-tailed recognition[J]. arXiv preprint arXiv:1910.09217, 2019.

#### 双线性池化方法

- [ ] Lin T Y, RoyChowdhury A, Maji S. Bilinear cnn models for fine-grained visual recognition[C]//Proceedings of the IEEE international conference on computer vision. 2015: 1449-1457

#### 细粒度分类

- [ ] WSDAN: Hu T, Qi H, Huang Q, et al. See better before looking closer: Weakly supervised data augmentation network for fine-grained visual classification[J]. arXiv preprint arXiv:1901.09891, 2019

#### 神经网络可视化

- [ ] Zhou B, Khosla A, Lapedriza A, et al. Learning deep features for discriminative localization[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2921-2929.
- [x] Selvaraju R R, Cogswell M, Das A, et al. Grad-cam: Visual explanations from deep networks via gradient-based localization[C]//Proceedings of the IEEE international conference on computer vision. 2017: 618-626.

#### 实验结果

| Backbone | Attn | Loss                      | Top1 Acc |
| -------- | ---- | ------------------------- | -------- |
| ResNet50 | None | label smothing softmax ce | 87.159%  |

