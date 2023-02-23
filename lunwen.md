主题 瑕疵检测与分割(AD&S) + 3D



# BACK TO THE FEATURE: CLASSICAL 3D FEATURES ARE  (ALMOST) ALL YOU NEED FOR 3D ANOMALY DETECTION



## Introduction



1. 现在，纯颜色方法如PatchCore比3D的瑕疵检测与分割方法要好，且好的程度大。
2. 因为有一些类型的瑕疵仅使用颜色信息是难以辨别出的 所以3D信息是潜在有用的。比如饼干和土豆上的凹陷。
3. 关于3D信息在对于瑕疵检测和分割的表示方法。此文章探索了人工和深度学习的表示方法。发现经典的，手工的3D点云描述符超过了现在其它的方法，包括基于学习的方法。
4. 因为颜色是有用的，比如格兰头的表面纹理的变形，泡沫板的颜色瑕疵，所以这篇文章采用颜色信息和3D信息混合的方法，并在MVTec数据集中获得了高的提升

## Related Work

1. 经典方法 KNN KDE GMM PCA 和 OCSVM 和 isolation forests
2. 借助深度学习 PCA to DAGMM. OCSVM to DeepSVDD
3. 一条创新的路线，限于瑕疵检测：使用自监督学习。有拓展RotNet的 有拓展传统方法的
4. 这篇文章采用另一条路线 使用预训练得到的特征 并将特征使用到KNN中，路线的例子有Perera and Patel和PANDA。并且这个方法也被拓展应用到瑕疵分割中如SPADE, PADIM和PatchCore. 
5. 最近有 在提取的特征上使用更先进的密度估计(density estimation)模型 的工作 例子是Fast Flow
6. 其它的瑕疵分割方法包括Student-Teacher autoencoder approches 和 自监督瑕疵生成如CutPaste 和 NSA



相比2D图像下次研究 3D 瑕疵检测并没有被深度研究。 有医学上使用体素的方法：在医学体素使用3D autoencoder方法. 体素与点云有较大区别。因为缺少用于3D点云的瑕疵检测数据集，所以有了MVTec 3D-AD。

并且还有基于3D 点云的瑕疵检测方法3D-ST。



## Problem Definition

训练用样本： x1-xN是正常的。测试时，给出测试样本y。

瑕疵检测的目标是：获得一个函数 如果函数输入是正常的样品 那么输出是小于等于零的。对应反之相反。

瑕疵分割的目标是：获得一个函数 输入样本y和像素i 如果样本y在像素i上是正常的 输出小于等于零。



如今顶尖方法步骤：

i) 提取local region的特征 

ii) 估计正常的local region的概率密度。比如PatchCore和SPADE使用与正常训练集之间的 nearest-neighbor distance来计算



特征：

local region 可能包括一个像素或多个像素。特征函数是输入样本和区域 得到特征。 这篇文章主要专注于特征这个步骤。目的是找到 学习的或者是人工的 对3DAD&S的 表示。



瑕疵给分：

训练时 给出所有训练样本的local region的特征。训练一个模型。这个模型（也就是瑕疵分割的目标）输入是 测试样本y，所有训练样本x,  所有训练样本中的local region j， 测试样本中的local regionj1。 输出是这个j1是瑕疵的可能性。

具体方法是通过 特征函数（测试样本y， j1） 和 特征函数（所有训练样本x， 所有j）得到 两个特征 通过计算这两个特征的 k-Nearest-Neighbor distance 的距离来计算可能性。



 k-Nearest-Neighbor distance 是 non-parametric approaches are much simpler and require no training. can be significantly sped up



3D 表示：

RGB图像缺少3D信息

深度图 有组织点云 无组织点云 体素。

有组织点云包含空间信息 所以可以被视为图像 运用到RGB图像的方法。无组织点云则需要另外特殊的方法

从点云中可以提取体素



Benchmark 测试：

介绍了AD-3D的请况 和 三种baselines GAN-based, Autoencoder-based, Variation Model (simple baseline based on per-pixel mean and standard deviation).  这些方法工作于深度图像，运用3D和颜色信息的 有着额外变量的体素空间





指标：

ROCAUC -> pixel-wise ROCAUC

PRO:



## 探索

### 如今3D方法超过2D方法吗：

3D方法选择Voxel GAN 体素-rgb-生成模型。 3D-ST方法

颜色方法选择PatchCore。

关于预训练，PatchCore使用ImageNet。 3D-ST使用ModelNet10预训练教师模型。





### 3D信息有用吗：

有一些瑕疵如凹陷小坑 土豆饼干

背景变化可能造成假阳性。



### 成功合适的3D表示和它们的关键性质：

对图片的表示：

基于学习的表示：

1. Depth-only ImageNet feature. Learning based.

2. NSA. Learning based. 瑕疵生成用这个

人工图片表示：

深度pattern比颜色pattern要简单 所以认为 手工的 简单的描述符就能满足 

1. raw depth value: 

2. HoG: 不受小变化位移和旋转的干扰，包含空间结构信息。但是会受旋转的影响，而且又不受瑕疵影响 所以couterproductive

3. Dense Scale-Invariant Feature Transform(D-SIFT) 不受旋转 缩放 位移的影响  

上面所说的都是基于深度的 包括了人工的和学习的 其中D-SIFT





3D的 不受旋转影响的表示：

1. Fast Point Feature Histograms(FPFH). 先计算kNN 得到各区域的中心点。  然后再算histogram。





对于点云的 基于学习的表示：

PointNeXt. U-Net 架构. encoder 抽象出feature decoder 插值

SpinNet.  不受旋转影响的表示，这是通过体素化实现的



上述的学习方法 并没有超过FPFH FPFH是旧的中最优的



### 结合颜色和3D有用吗

BTF表示 用image-net提取颜色feature，用FPFH提取3D的feature。 将这两个feature拼接起来得到BTF。



### 实现细节

3D 点云预处理

去除背景 使用 RANSAC 去除outliers connected-components-based算法。 这个算法大提升深度方法 小降3D方法。







# One-Class Classification of 3D Point Clouds Using Dynamic Graph CNN





point cloud dataset: 

ModelNet40

Completion3D

Dex-Net 2.0

Multi-View Partial point cloud

ShapeNet-ViPC

ShapeNet





anomaly detection 

Reconstruction-based methods

embedding-based methods



Feature Extraction from point cloud

 multivariate Gaussian distribution of Normal data



Mahalanobis distance



# Towards Total Recall in Industrial Anomaly Detection

## Introduction



cold-start: train with only nominal data. Out of distribution detection problem.



瑕疵有大有小 种类繁多 所以工业瑕疵分类是困难的。



已有的cold-start工作又用autoencoding和GANs的 以及其它unsupervised的adaptation methods。

最近有使用 来源于ImageNet分类的 没有对目标分布进行adaptation的 深度表示



这些方法缺少适应性 但AD&S做得好



基本方法是比较测试样本和正常样本。使用多尺度深度特征表示。

细粒 不易察觉得 可以用高分辨率特征表示 结构变位和全图瑕疵检测需要更将抽象概括得等级。

这样的方法是缺少适应性 因为高抽象级别的matching confidence收到限制 

从ImageNet而来的高层次抽象特征与工业环境符合较少

并且这些方法在测试时 因为较少数量的可执行高级别特征表示 对正常样本的信息利用较少





PatchCore做的事：

1. 在测试时 使用更多的正常样本信息。
2. 减少倾向于ImageNet类别的偏移
3. 保持高响应速度

使用本地集成的，中间层网络patch特征是。这样尽量少的使用ImageNet高分辨率类别的bias。同时在本地邻居的特征集成保留了猪狗的空间信息。

这个做法产生了大的memory bank来利用正常内容的信息。

最后 为了减少响应时间和内存占用， 会对正常样本的特征banks进行subsample







## Related Works

大多数异常检测采用学习正常样本的特征表示。这个可以通过autoencoder模型来实现。为了更好地估计正常样本的特征分布，会将模型拓展到Gaussian mixture模型、generative adversarial training objectives、

invariance towards predefined physical aug-mentations、 

robustness of hidden features to reintroduction of reconstructions、

prototypical memory banks、

attention-guidance、 

structural objectives
或者constrained representation spaces

还有其它无监督学习方法如GAN, 学习预测预先定义的几何变化 或者通过normalizing flows。





在得到各个正常样本的表示 和测试样本的表示后，瑕疵检测可以变为重建误差， 与k个最近邻居的距离，或者one-class分类。在上面这些方法中，瑕疵定位可以通过每像素重建误差和基于显著性的方法来实现。







使用预训练模型，比如一些在ImageNet上的训练，不做任何适应性训练，在过去达到了最好的成果。这些借助预训练的方法有SPADE, kNN-based 方法 以及 PaDim  使用bag-of-feature方法 估计patch级别的特征表示分布并计算mahalanobis 距离。 还有为了从训练自然预训练模型到工业图片数据，进行学生教师的蒸馏，或者normalize flows的工作.





PatchCode 有使用SPADE和PaDiM. 

SPADE使用正常样本的特征的memory-bank，这些特征从预训练的backbone网络中提取。并且这些网络有用于像素级别的瑕疵检测和图片的瑕疵检测。

PatchCore也使用memory bank，但是使用neighborhood-aware patch-level的特征。这样更多的正常样本内容被保存，better fitting inductive bias is incorporated。

Coresets 在一些深度学习方法中被使用并取得了成果。

PatchCore的瑕疵检测和分割是patch-level的，这些都与Padim的工作有关，在测试时，对每个patch都可以使用patch-feature的memory bank。 这点是在Padim上更加拓宽的。这样PatchCore可以更不依赖图片对其并且使用更多正常样本信息。



## Method



































































