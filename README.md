* 推荐阅读：<br>
  * [ICCV2021/2019/2017 论文/代码/解读/直播合集](https://github.com/extreme-assistant/ICCV2021-Paper-Code-Interpretation)
  * [2020-2021年计算机视觉综述论文汇总](https://github.com/extreme-assistant/survey-computer-vision)
  * [国内外优秀的计算机视觉团队汇总](https://github.com/extreme-assistant/Awesome-CV-Team)
------

# CVPR2022最新信息及论文下载（Papers/Codes/Project/PaperReading／Demos/直播分享／论文分享会等）

官网链接：http://CVPR2022.thecvf.com<br>
时间：2021年6月19日-6月24日<br>
论文接收公布时间：2022年3月2日<br>

相关问题：[如何评价 CVPR2022 的论文接收结果？](https://www.zhihu.com/question/519162597)<br>
相关报道：[CVPR 2022 接收结果出炉！录用 2067 篇，接收数量上升24%](https://mp.weixin.qq.com/s/hAWrCpili4CICJzKrQ8Sog)<br>


>update: <br>
>2022/3/3 [更新 19 篇](https://bbs.cvmart.net/articles/6149)<br>
>2022/3/4 [更新 29 篇](https://bbs.cvmart.net/articles/6151)<br>
>2022/3/7 [更新 17 篇](https://bbs.cvmart.net/articles/6158)<br>
>2022/3/9 [更新 57 篇](https://bbs.cvmart.net/articles/6164)<br>
>2022/3/10 [更新 8 篇](https://bbs.cvmart.net/articles/6171)<br>
>2022/3/11 [更新 18 篇](https://bbs.cvmart.net/articles/6173)<br>
>2022/3/14 [更新 11 篇](https://bbs.cvmart.net/articles/6175)<br>
>2022/3/15 [更新 30 篇](https://bbs.cvmart.net/articles/6178)<br>
>2022/3/16 [更新 16 篇](https://bbs.cvmart.net/articles/6181)<br>
>2022/3/17 [更新 24 篇](https://bbs.cvmart.net/articles/6185)<br>
>2022/3/18 [更新 25 篇](https://bbs.cvmart.net/articles/6192)<br>
>2022/3/22 [更新 52 篇](https://bbs.cvmart.net/articles/6204)<br>
>2022/3/23 [更新 29 篇](https://bbs.cvmart.net/articles/6206)<br>
>2022/3/24 [更新 22 篇](https://bbs.cvmart.net/articles/6209)<br>
>2022/3/25 [更新 29 篇](https://bbs.cvmart.net/articles/6213)<br>
>2022/3/28 [更新 29 篇](https://bbs.cvmart.net/articles/6222)

<br><br>

# 目录

[1. CVPR2022 接受论文/代码分方向汇总（更新中）](#1)<br>
[2. CVPR2022 Oral（更新中）](#2)<br>
[3. CVPR2022 论文解读汇总（更新中）](#3)<br>
[4. CVPR2022 极市论文分享](#4)<br>
[5. To do list](#5)<br>

<br>

<a name="1"/> 

# 1.CVPR2022接受论文/代码分方向整理(持续更新)


## 分类目录：

### [1. 检测](#detection)
* [2D目标检测(2D Object Detection)](#IOD)
* [视频目标检测(Video Object Detection)](#VOD)
* [3D目标检测(3D Object Detection)](#3DOD)
* [人物交互检测(HOI Detection)](#HOI)
* [伪装目标检测(Camouflaged Object Detection)](#COD)
* [旋转目标检测(Rotation Object Detection)](#ROD)
* [显著性目标检测(Saliency Object Detection)](#SOD)
* [关键点检测(Keypoint Detection)](#KeypointDetection)
* [车道线检测(Lane Detection)](#LaneDetection)
* [边缘检测(Edge Detection)](#EdgeDetection)
* [消失点检测(Vanishing Point Detection)](#VPD)
* [异常检测(Anomaly Detection)](#AnomalyDetection)

### [2. 分割(Segmentation)](#Segmentation)
* [图像分割(Image Segmentation)](#ImageSegmentation)
* [全景分割(Panoptic Segmentation)](#PanopticSegmentation)
* [语义分割(Semantic Segmentation)](#SemanticSegmentation)
* [实例分割(Instance Segmentation)](#InstanceSegmentation)
* [超像素(Superpixel)](#Superpixel)
* [视频目标分割(Video Object Segmentation)](#VOS)
* [抠图(Matting)](#Matting)
* [密集预测(Dense Prediction)](#DensePrediction)

### [3. 图像处理(Image Processing)](#ImageProcessing)

* [超分辨率(Super Resolution)](#SuperResolution)
* [图像复原/图像增强/图像重建(Image Restoration/Image Reconstruction)](#ImageRestoration)
* [图像去阴影/去反射(Image Shadow Removal/Image Reflection Removal)](#ISR)
* [图像去噪/去模糊/去雨去雾(Image Denoising)](#ImageDenoising)
* [图像编辑/图像修复(Image Edit/Image Inpainting)](#ImageEdit)
* [图像翻译(Image Translation)](#ImageTranslation)
* [图像质量评估(Image Quality Assessment)](#IQA)
* [风格迁移(Style Transfer)](#StyleTransfer)

### [4. 视频处理(Video Processing)](#VideoProcessing)
* [视频编辑(Video Editing)](#VideoEditing)
* [视频生成/视频合成(Video Generation/Video Synthesis)](#VideoGeneration)

### [5. 估计(Estimation)](#Estimation)
* [光流/运动估计(Flow/Motion Estimation)](#Flow/Pose/MotionEstimation)
* [深度估计(Depth Estimation)](#DepthEstimation)
* [人体解析/人体姿态估计(Human Parsing/Human Pose Estimation)](#HumanPoseEstimation)
* [手势估计(Gesture Estimation)](#GestureEstimation)

### [6. 图像&视频检索/视频理解(Image&Video Retrieval/Video Understanding)](#ImageRetrieval)
* [行为识别/行为识别/动作识别/检测/分割(Action/Activity Recognition)](#ActionRecognition)
* [行人重识别/检测(Re-Identification/Detection)](#Re-Identification)
* [图像/视频字幕(Image/Video Caption)](#VideoCaption)

### [7. 人脸(Face)](#Face)
* [人脸识别/检测(Facial Recognition/Detection)](#FacialRecognition)
* [人脸生成/合成/重建/编辑(Face Generation/Face Synthesis/Face Reconstruction/Face Editing)](#FaceSynthesis)
* [人脸伪造/反欺骗(Face Forgery/Face Anti-Spoofing)](#FaceAnti-Spoofing)

### [8. 三维视觉(3D Vision)](#3DVision)
* [点云(Point Cloud)](#3DPC)
* [三维重建(3D Reconstruction)](#3DReconstruction)
* [场景重建/视图合成/新视角合成(Novel View Synthesis)](#NeRF)

### [9. 目标跟踪(Object Tracking)](#ObjectTracking)

### [10. 医学影像(Medical Imaging)](#MedicalImaging)

### [11. 文本检测/识别/理解(Text Detection/Recognition/Understanding)](#TDR)

### [12. 遥感图像(Remote Sensing Image)](#RSI)

### [13. GAN/生成式/对抗式(GAN/Generative/Adversarial)](#GAN)

### [14. 图像生成/图像合成(Image Generation/Image Synthesis)](#IGIS)

### [15. 场景图(Scene Graph](#SG)
* [场景图生成(Scene Graph Generation)](#SGG)
* [场景图预测(Scene Graph Prediction)](#SGP)
* [场景图理解(Scene Graph Understanding)](#SGU)

### [16. 视觉定位/位姿估计(Visual Localization/Pose Estimation)](#VisualLocalization)

### [17. 视觉推理/视觉问答(Visual Reasoning/VQA)](#VisualReasoning)

### [18. 视觉预测(Vision-based Prediction)](#Vision-basedPrediction)

### [19. 神经网络结构设计(Neural Network Structure Design)](#NNS)
* [CNN](#CNN)
* [Transformer](#Transformer)
* [图神经网络(GNN)](#GNN)
* [神经网络架构搜索(NAS)](#NAS)
* [MLP](#MLP)

### [20. 神经网络可解释性(Neural Network Interpretability)](#interpretability)

### [21. 数据集(Dataset)](#Dataset)

### [22. 数据处理(Data Processing)](#DataProcessing)
* [数据增广(Data Augmentation)](#DataAugmentation)
* [归一化/正则化(Batch Normalization)](#BatchNormalization)
* [图像聚类(Image Clustering)](#ImageClustering)
* [图像压缩(Image Compression)](#ImageCompression)

### [23. 图像特征提取与匹配(Image feature extraction and matching)](#matching)

### [24. 视觉表征学习(Visual Representation Learning)](#VisualRL)

### [25. 模型训练/泛化(Model Training/Generalization)](#ModelTraining)
* [噪声标签(Noisy Label)](#NoisyLabel)
* [长尾分布(Long-Tailed Distribution)](#Long-Tailed)

### [26. 模型压缩(Model Compression)](#ModelCompression)
* [知识蒸馏(Knowledge Distillation)](#KnowledgeDistillation)
* [剪枝(Pruning)](#Pruning)
* [量化(Quantization)](#Quantization)

### [27. 模型评估(Model Evaluation)](#ModelEvaluation)

### [28. 图像分类(Image Classification)](#ImageClassification)

### [29. 图像计数(Image Counting)](#CrowdCounting)

### [30. 机器人(Robotic)](#Robotic)

### [31. 自监督学习/半监督学习/无监督学习(Self-supervised Learning/Semi-supervised Learning)](#self-supervisedlearning)

### [32. 多模态学习(Multi-Modal Learning)](#MMLearning)
* [视听学习(Audio-visual Learning)](#Audio-VisualLearning)
* [视觉-语言（Vision-language）](#VLRL)

### [33. 主动学习(Active Learning)](#ActiveLearning)

### [34. 小样本学习/零样本学习(Few-shot/Zero-shot Learning)](#Few-shotLearning)

### [35. 持续学习(Continual Learning/Life-long Learning)](#ContinualLearning)

### [36. 迁移学习/domain/自适应(Transfer Learning/Domain Adaptation)](#domain)

### [37. 度量学习(Metric Learning)](#MetricLearning)

### [38. 对比学习(Contrastive Learning)](#ContrastiveLearning)

### [39. 增量学习(Incremental Learning)](#IncrementalLearning)

### [40. 强化学习(Reinforcement Learning)](#RL)

### [41. 元学习(Meta Learning)](#MetaLearning)

### [42. 联邦学习(Federated Learning](#federatedlearning)






### [其他](#100)



<br><br>

<a name="detection"/> 

## 检测



<br>

<a name="IOD"/> 

### 2D目标检测(2D Object Detection)

[14] QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection(用于加速高分辨率小目标检测的级联稀疏查询)<br>
[paper](https://arxiv.org/abs/2103.09136) | [code](https://github.com/ChenhongyiYang/QueryDet-PyTorch)<br><br>

[13] End-to-End Human-Gaze-Target Detection with Transformers(使用 Transformer 进行端到端的人眼目标检测)<br>
[paper](https://arxiv.org/abs/2203.10433)<br><br>

[12] Progressive End-to-End Object Detection in Crowded Scenes(拥挤场景中的渐进式端到端对象检测)<br>
[paper](https://arxiv.org/abs/2203.07669) | [code](https://github.com/megvii-model/Iter-E2EDET)<br><br>

[11] Real-time Object Detection for Streaming Perception(用于流感知的实时对象检测)<br>
[paper](https://arxiv.org/abs/2203.12338) | [code](https://github.com/yancie-yjr/StreamYOLO)<br><br>

[10] Oriented RepPoints for Aerial Object Detection(面向空中目标检测的 RepPoints)(**小目标检测**)<br>
[paper](https://arxiv.org/abs/2105.11111) | [code](https://github.com/LiWentomng/OrientedRepPoints)<br><br>

[9] Confidence Propagation Cluster: Unleash Full Potential of Object Detectors(信心传播集群：释放物体检测器的全部潜力)<br>
[paper](https://arxiv.org/abs/2112.00342)<br><br>

[8] Semantic-aligned Fusion Transformer for One-shot Object Detection(用于一次性目标检测的语义对齐融合转换器)<br>
[paper](https://arxiv.org/abs/2203.09093)<br><br>

[7] A Dual Weighting Label Assignment Scheme for Object Detection(一种用于目标检测的双重加权标签分配方案)<br>
[paper](https://arxiv.org/abs/2203.09730) | [code](https://github.com/strongwolf/DW)<br><br>

[6] MUM : Mix Image Tiles and UnMix Feature Tiles for Semi-Supervised Object Detection(混合图像块和 UnMix 特征块用于半监督目标检测)<br>
[paper](https://arxiv.org/abs/2111.10958) | [code](https://github.com/JongMokKim/mix-unmix)<br><br>

[5] SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection(域自适应对象检测的语义完全图匹配)<br>
[paper](https://arxiv.org/abs/2203.06398) | [code](https://github.com/CityU-AIM-Group/SIGMA)<br><br>

[4] Accelerating DETR Convergence via Semantic-Aligned Matching(通过语义对齐匹配加速 DETR 收敛)<br>
[paper](https://arxiv.org/abs/2203.06883) | [code](https://github.com/ZhangGongjie/SAM-DETR)<br><br>

[3] Focal and Global Knowledge Distillation for Detectors(探测器的焦点和全局知识蒸馏)<br>
keywords: Object Detection, Knowledge Distillation<br>
[paper](https://arxiv.org/abs/2111.11837) | [code](https://github.com/yzd-v/FGD)<br><br>

[2] Unknown-Aware Object Detection: Learning What You Don't Know from Videos in the Wild(未知感知对象检测：从野外视频中学习你不知道的东西)<br>
[paper](https://arxiv.org/abs/2203.03800) | [code](https://github.com/deeplearning-wisc/stud)<br><br>

[1] Localization Distillation for Dense Object Detection(密集对象检测的定位蒸馏)<br>
keywords: Bounding Box Regression, Localization Quality Estimation, Knowledge Distillation<br>
[paper](https://arxiv.org/abs/2102.12252) | [code](https://github.com/HikariTJU/LD)<br>
解读：[南开程明明团队和天大提出LD：目标检测的定位蒸馏](https://zhuanlan.zhihu.com/p/474955539)<br><br>

<br>


<a name="VOD"/> 

### 视频目标检测(Video Object Detection)

[1] Unsupervised Activity Segmentation by Joint Representation Learning and Online Clustering(通过联合表示学习和在线聚类进行无监督活动分割)<br>
[paper](https://arxiv.org/abs/2105.13353) | [video](https://www.youtube.com/watch?v=i4Fh_3nzzUI&t=12s)<br><br>

<br>

<a name="3DOD"/> 

### 3D目标检测(3D object detection)


[15] Point2Seq: Detecting 3D Objects as Sequences(将 3D 对象检测为序列)<br>
[paper](https://arxiv.org/abs/2203.13394) | [code](https://github.com/ocNflag/point2seq)<br><br>

[14] MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection(用于单目 3D 对象检测的深度感知transformer)<br>
[paper](https://arxiv.org/abs/2203.13310) | [code](https://github.com/ZrrSkywalker/MonoDETR.git)<br><br>

[13] TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers(用于 3D 对象检测的稳健 LiDAR-Camera Fusion 与 Transformer)<br>
[paper](https://arxiv.org/abs/2203.11496) | [code](https://github.com/XuyangBai/TransFusion)<br><br>

[12] Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds(学习用于 3D LiDAR 点云的高效基于点的检测器)<br>
[paper](https://arxiv.org/abs/2203.11139) | [code](https://github.com/yifanzhang713/IA-SSD)<br><br>

[11] Sparse Fuse Dense: Towards High Quality 3D Detection with Depth Completion(迈向具有深度完成的高质量 3D 检测)<br>
[paper](https://arxiv.org/abs/2203.09780)<br><br>

[10] MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer(使用深度感知 Transformer 的单目 3D 对象检测)<br>
[paper](https://arxiv.org/abs/2203.10981) | [code](https://github.com/kuanchihhuang/MonoDTR)<br><br>

[9] Voxel Set Transformer: A Set-to-Set Approach to 3D Object Detection from Point Clouds(从点云进行 3D 对象检测的 Set-to-Set 方法)<br>
[paper](https://arxiv.org/abs/2203.10314) | [code](https://github.com/skyhehe123/VoxSeT)<br><br>

[8] VISTA: Boosting 3D Object Detection via Dual Cross-VIew SpaTial Attention<br>
[paper](https://arxiv.org/abs/2203.09704) | [code](https://github.com/Gorilla-Lab-SCUT/VISTA)<br><br>

[7] MonoJSG: Joint Semantic and Geometric Cost Volume for Monocular 3D Object Detection(单目 3D 目标检测的联合语义和几何成本量)<br>
[paper](https://arxiv.org/abs/2203.08563) | [code](https://github.com/lianqing11/MonoJSG)<br><br>

[6] DeepFusion: Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection(用于多模态 3D 目标检测的激光雷达相机深度融合)<br>
[paper](https://arxiv.org/abs/2203.08195) | [code](https://github.com/tensorflow/lingvo/tree/master/lingvo/)<br><br>

[5] Point Density-Aware Voxels for LiDAR 3D Object Detection(用于 LiDAR 3D 对象检测的点密度感知体素)<br>
[paper](https://arxiv.org/abs/2203.05662) | [code](https://github.com/TRAILab/PDV)<br><br>

[4] Back to Reality: Weakly-supervised 3D Object Detection with Shape-guided Label Enhancement(带有形状引导标签增强的弱监督 3D 对象检测)<br>
[paper](https://arxiv.org/abs/2203.05238) | [code](https://github.com/xuxw98/BackToReality)<br><br>

[3] Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes(在 3D 场景中实现稳健的定向边界框检测)<br>
[paper](https://arxiv.org/abs/2011.12001) | [code](https://github.com/qq456cvb/CanonicalVoting)<br><br>

[2] A Versatile Multi-View Framework for LiDAR-based 3D Object Detection with Guidance from Panoptic Segmentation(在全景分割的指导下，用于基于 LiDAR 的 3D 对象检测的多功能多视图框架)<br>
keywords: 3D Object Detection with Point-based Methods, 3D Object Detection with Grid-based Methods, Cluster-free 3D Panoptic Segmentation, CenterPoint 3D Object Detection<br>
[paper](https://arxiv.org/abs/2203.02133)<br><br>

[1] Pseudo-Stereo for Monocular 3D Object Detection in Autonomous Driving(自动驾驶中用于单目 3D 目标检测的伪立体)<br>
keywords: Autonomous Driving, Monocular 3D Object Detection<br>
[paper](https://arxiv.org/abs/2203.02112) | [code](https://github.com/revisitq/Pseudo-Stereo-3D)<br><br>

<br>

<a name="HOI"/> 

### 人物交互检测(HOI Detection)

<br>

<a name="COD"/> 

### 伪装目标检测(Camouflaged Object Detection)

[2] Implicit Motion Handling for Video Camouflaged Object Detection(视频伪装对象检测的隐式运动处理)<br>
[paper](https://arxiv.org/abs/2203.07363) | [dataset](https://xueliancheng.github.io/SLT-Net-project)<br><br>

[1] Zoom In and Out: A Mixed-scale Triplet Network for Camouflaged Object Detection(放大和缩小：用于伪装目标检测的混合尺度三元组网络)<br>
[paper](https://arxiv.org/abs/2203.02688) | [code](https://github.com/lartpang/ZoomNet)<br><br>

<br>

<a name="ROD"/> 

### 旋转目标检测(Rotation Object Detection)

<br>

<a name="SOD"/> 

### 显著性目标检测(Saliency Object Detection)

[2] Bi-directional Object-context Prioritization Learning for Saliency Ranking(显着性排名的双向对象上下文优先级学习)<br>
[paper](https://arxiv.org/abs/2203.09416) | [code](https://github.com/GrassBro/OCOR)<br><br>

[1] Democracy Does Matter: Comprehensive Feature Mining for Co-Salient Object Detection()<br>
[paper](https://arxiv.org/abs/2203.05787)<br><br>


<br>

<a name="KeypointDetection"/> 

### 关键点检测(Keypoint Detection)

[1] UKPGAN: A General Self-Supervised Keypoint Detector(一个通用的自监督关键点检测器)<br>
[paper](https://arxiv.org/abs/2011.11974) | [code](https://github.com/qq456cvb/UKPGAN)<br><br>

<br>

<a name="LaneDetection"/> 

### 车道线检测(Lane Detection)

[2] CLRNet: Cross Layer Refinement Network for Lane Detection(用于车道检测的跨层细化网络)<br>
[paper](https://arxiv.org/abs/2203.10350)<br><br>

[1] Rethinking Efficient Lane Detection via Curve Modeling(通过曲线建模重新思考高效车道检测)<br>
keywords: Segmentation-based Lane Detection, Point Detection-based Lane Detection, Curve-based Lane Detection, autonomous driving<br>
[paper](https://arxiv.org/abs/2203.02431) | [code](https://github.com/voldemortX/pytorch-auto-drive)<br><br>

<br>

<a name="EdgeDetection"/> 

### 边缘检测(Edge Detection)

[1] EDTER: Edge Detection with Transformer(使用transformer的边缘检测)<br>
[paper](https://arxiv.org/abs/2203.08566) | [code](https://github.com/MengyangPu/EDTER)<br><br>

<br>

<a name="VPD"/> 

### 消失点检测(Vanishing Point Detection)

[1] Deep vanishing point detection: Geometric priors make dataset variations vanish(深度**消失点检测**：几何先验使数据集变化消失)<br>
[paper](https://arxiv.org/abs/2203.08586) | [code](https://github.com/yanconglin/VanishingPoint_HoughTransform_GaussianSphere)<br><br>

<br>

<a name="AnomalyDetection"/> 

### 异常检测(Anomaly Detection)

[4] UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection(监督开放集视频异常检测的新基准)<br>
[paper](https://arxiv.org/abs/2111.08644) | [code](https://github.com/lilygeorgescu/UBnormal)<br><br>

[3] ViM: Out-Of-Distribution with Virtual-logit Matching(具有虚拟 logit 匹配的分布外)(**OOD检测**)<br>
[paper](https://arxiv.org/abs/2203.10807) | [code](https://github.com/haoqiwang/vim)<br><br>

[2] Generative Cooperative Learning for Unsupervised Video Anomaly Detection(用于无监督视频异常检测的生成式协作学习)<br>
[paper](https://arxiv.org/abs/2203.03962)<br><br>

[1] Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection(用于异常检测的自监督预测卷积注意力块)(论文暂未上传)<br>
[paper](https://arxiv.org/abs/2111.09099) | [code](https://github.com/ristea/sspcab)<br><br>


<br>

<a name="Segmentation"/> 


## 分割(Segmentation)

<br>

<a name="ImageSegmentation"/> 

### 图像分割(Image Segmentation)

[3] Learning What Not to Segment: A New Perspective on Few-Shot Segmentation(学习不分割的内容：关于小样本分割的新视角)<br>
[paper](https://arxiv.org/abs/2203.07615) | [code](http://github.com/chunbolang/BAM)<br><br>

[2] CRIS: CLIP-Driven Referring Image Segmentation(CLIP 驱动的参考图像分割)<br>
[paper](https://arxiv.org/abs/2111.15174)<br><br>

[1] Hyperbolic Image Segmentation(双曲线图像分割)<br>
[paper](https://arxiv.org/abs/2203.05898)<br><br>

<br>

<a name="PanopticSegmentation"/> 

### 全景分割(Panoptic Segmentation)

[2] Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers(使用 Transformers 深入研究全景分割)<br>
[paper](https://arxiv.org/abs/2109.03814) | [code](https://github.com/zhiqi-li/Panoptic-SegFormer)<br><br>

[1] Bending Reality: Distortion-aware Transformers for Adapting to Panoramic Semantic Segmentation(弯曲现实：适应全景语义分割的失真感知Transformer)<br>
keywords: Semantic- and panoramic segmentation, Unsupervised domain adaptation, Transformer<br>
[paper](https://arxiv.org/abs/2203.01452) | [code](https://github.com/jamycheung/Trans4PASS)<br><br>

<br>

<a name="SemanticSegmentation"/> 

### 语义分割(Semantic Segmentation)

[16] Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation(半监督语义分割的扰动和严格均值教师)<br>
[paper](https://arxiv.org/abs/2111.12903)<br><br>

[15] Class-Balanced Pixel-Level Self-Labeling for Domain Adaptive Semantic Segmentation(用于域自适应语义分割的类平衡像素级自标记)<br>
[paper](https://arxiv.org/abs/2203.09744) | [code](https://github.com/lslrh/CPSL)<br><br>

[14] Regional Semantic Contrast and Aggregation for Weakly Supervised Semantic Segmentation(弱监督语义分割的区域语义对比和聚合)<br>
[paper](https://arxiv.org/abs/2203.09653) | [code](https://github.com/maeve07/RCA.git)<br><br>

[13] Tree Energy Loss: Towards Sparsely Annotated Semantic Segmentation(走向稀疏注释的语义分割)<br>
[paper](https://arxiv.org/abs/2203.10739) | [code](https://github.com/megviiresearch/TEL)<br><br>

[12] Scribble-Supervised LiDAR Semantic Segmentation<br>
[paper](https://arxiv.org/abs/2203.08537) |[code](http://github.com/ouenal/scribblekitti)<br><br>

[11] ADAS: A Direct Adaptation Strategy for Multi-Target Domain Adaptive Semantic Segmentation(多目标域自适应语义分割的直接适应策略)<br>
[paper](https://arxiv.org/abs/2203.06811)<br><br>

[10] Weakly Supervised Semantic Segmentation by Pixel-to-Prototype Contrast(通过像素到原型对比的弱监督语义分割)<br>
[paper](https://arxiv.org/abs/2110.07110)<br><br>

[9] Representation Compensation Networks for Continual Semantic Segmentation(连续语义分割的表示补偿网络)<br>
[paper](https://arxiv.org/abs/2203.05402) | [code](https://github.com/zhangchbin/RCIL)<br><br>

[8] Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels(使用不可靠伪标签的半监督语义分割)<br>
[paper](https://arxiv.org/abs/2203.03884) | [code](https://github.com/Haochen-Wang409/U2PL/) | [project](https://haochen-wang409.github.io/U2PL/)<br><br>

[7] Weakly Supervised Semantic Segmentation using Out-of-Distribution Data(使用分布外数据的弱监督语义分割)<br>
[paper](https://arxiv.org/abs/2203.03860) | [code](https://github.com/naver-ai/w-ood)<br><br>

[6] Self-supervised Image-specific Prototype Exploration for Weakly Supervised Semantic Segmentation(弱监督语义分割的自监督图像特定原型探索)<br>
[paper](https://arxiv.org/abs/2203.02909) | [code](https://github.com/chenqi1126/SIPE)<br><br>

[5] Multi-class Token Transformer for Weakly Supervised Semantic Segmentation(用于弱监督语义分割的多类token Transformer)<br>
[paper](https://arxiv.org/abs/2203.02891) | [code](https://github.com/xulianuwa/MCTformer)<br><br>

[4] Cross Language Image Matching for Weakly Supervised Semantic Segmentation(用于弱监督语义分割的跨语言图像匹配)<br>
[paper](https://arxiv.org/abs/2203.02668)<br><br>

[3] Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers(从注意力中学习亲和力：使用 Transformers 的端到端弱监督语义分割)<br>
[paper](https://arxiv.org/abs/2203.02664) | [code](https://github.com/rulixiang/afa)<br><br>

[2] ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation(让自我训练更好地用于半监督语义分割)<br>
keywords: Semi-supervised learning, Semantic segmentation, Uncertainty estimation<br>
[paper](https://arxiv.org/abs/2106.05095) | [code](https://github.com/LiheYoung/ST-PlusPlus)<br><br>

[1] Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation(弱监督语义分割的类重新激活图)<br>
[paper](https://arxiv.org/pdf/2203.00962.pdf) | [code](https://github.com/zhaozhengChen/ReCAM)<br><br>

<br>

<a name="InstanceSegmentation"/> 

### 实例分割(Instance Segmentation)

[9] Noisy Boundaries: Lemon or Lemonade for Semi-supervised Instance Segmentation?(嘈杂的边界：半监督实例分割的柠檬还是柠檬水？)<br>
[paper](https://arxiv.org/abs/2203.13427)<br><br>

[8] SharpContour: A Contour-based Boundary Refinement Approach for Efficient and Accurate Instance Segmentation(一种用于高效准确实例分割的基于轮廓的边界细化方法)<br>
[paper](https://arxiv.org/abs/2203.13312) | [project](https://xyzhang17.github.io/SharpContour/)<br><br>

[7] Sparse Instance Activation for Real-Time Instance Segmentation(实时实例分割的稀疏实例激活)<br>
[paper](https://arxiv.org/abs/2203.12827) | [code](https://github.com/hustvl/SparseInst)<br><br>

[6] Mask Transfiner for High-Quality Instance Segmentation(用于高质量实例分割的 Mask Transfiner)<br>
[paper](https://arxiv.org/abs/2111.13673) | [code](https://github.com/SysCV/transfiner)<br><br>

[5] ContrastMask: Contrastive Learning to Segment Every Thing(对比学习分割每件事)<br>
[paper](https://arxiv.org/abs/2203.09775)<br><br>

[4] Discovering Objects that Can Move(发现可以移动的物体)<br>
[paper](https://arxiv.org/abs/2203.10159) | [code](https://github.com/zpbao/Discovery_Obj_Move/)<br><br>

[3] E2EC: An End-to-End Contour-based Method for High-Quality High-Speed Instance Segmentation(一种基于端到端轮廓的高质量高速实例分割方法)<br>
[paper](https://arxiv.org/abs/2203.04074) | [code](https://github.com/zhang-tao-whu/e2ec)<br><br>

[2] Efficient Video Instance Segmentation via Tracklet Query and Proposal(通过 Tracklet Query 和 Proposal 进行高效的视频实例分割)<br>
[paper](https://arxiv.org/abs/2203.01853)<br><br>

[1] SoftGroup for 3D Instance Segmentation on Point Clouds(用于点云上的 3D 实例分割)<br>
keywords: 3D Vision, Point Clouds, Instance Segmentation<br>
[paper](https://arxiv.org/abs/2203.01509) | [code](https://github.com/thangvubk/SoftGroup.git)<br><br>

<br>

<a name="Superpixel"/> 

### 超像素(Superpixel)

<br>

<a name="VOS"/> 

### 视频目标分割(Video Object Segmentation)

[1] Language as Queries for Referring Video Object Segmentation(语言作为引用视频对象分割的查询)<br>
[paper](https://arxiv.org/abs/2201.00487) | [code](https://github.com/wjn922/ReferFormer)<br><br>

<br>

<a name="Matting"/> 

### 抠图(Matting)

<br>

<a name="DensePrediction"/> 

### 密集预测(Dense Prediction)

[1] DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting(具有上下文感知提示的语言引导密集预测)<br>
[paper](https://arxiv.org/abs/2112.01518) | [code](https://github.com/raoyongming/DenseCLIP)<br><br>

<br>

<a name="VideoProcessing"/> 

## 视频处理(Video Processing)

[2] Unifying Motion Deblurring and Frame Interpolation with Events(将运动去模糊和帧插值与事件统一起来)<br>
[paper](https://arxiv.org/abs/2203.12178)<br><br>

[1] Neural Compression-Based Feature Learning for Video Restoration(用于视频复原的基于神经压缩的特征学习)<br>
[paper](https://arxiv.org/abs/2203.09208)<br><br>

<br>

<a name="VideoEditing"/> 

### 视频编辑(Video Editing)

[1] M3L: Language-based Video Editing via Multi-Modal Multi-Level Transformers(M3L：通过多模式多级transformer进行基于语言的视频编辑)<br>
[paper](https://arxiv.org/abs/2104.01122)<br><br>

<br>

<a name="VideoGeneration"/> 

### 视频生成/视频合成(Video Generation/Video Synthesis)

[2] Depth-Aware Generative Adversarial Network for Talking Head Video Generation(用于说话头视频生成的深度感知生成对抗网络)<br>
[paper](https://arxiv.org/abs/2203.06605) | [code](https://github.com/harlanhong/CVPR2022-DaGAN)<br><br>

[1] Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning(告诉我什么并告诉我如何：通过多模式调节进行视频合成)<br>
[paper](https://arxiv.org/abs/2203.02573) | [code](https://github.com/snap-research/MMVID)<br><br>

<br>

<a name="Estimation"/> 

## 估计(Estimation)


<br>

<a name="Flow/Pose/MotionEstimation"/> 

### 光流/运动估计(Optical Flow/Motion Estimation)

[2] Global Matching with Overlapping Attention for Optical Flow Estimation(具有重叠注意力的全局匹配光流估计)<br>
[paper](https://arxiv.org/abs/2203.11335) | [code](https://github.com/xiaofeng94/GMFlowNet)<br><br>

[1] CamLiFlow: Bidirectional Camera-LiDAR Fusion for Joint Optical Flow and Scene Flow Estimation(用于联合光流和场景流估计的双向相机-LiDAR 融合)<br>
[paper](https://arxiv.org/abs/2111.10502)<br><br>

<br>

<a name="DepthEstimation"/> 

### 深度估计(Depth Estimation)

[13] LGT-Net: Indoor Panoramic Room Layout Estimation with Geometry-Aware Transformer Network(具有几何感知变压器网络的室内全景房间布局估计)(布局估计)<br>
[paper](https://arxiv.org/abs/2203.01824) | [code](https://github.com/zhigangjiang/LGT-Net)<br><br>

[12] Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation(基于自适应相关的级联循环网络的实用立体匹配)<br>
[paper](https://arxiv.org/abs/2203.11483) | [project](https://github.com/megvii-research/CREStereo)<br><br>

[11] Depth Estimation by Combining Binocular Stereo and Monocular Structured-Light(结合双目立体和单目结构光的深度估计)<br>
[paper](https://arxiv.org/abs/2203.10493) | [code](https://github.com/YuhuaXu/MonoStereoFusion)<br><br>

[10] RGB-Depth Fusion GAN for Indoor Depth Completion(用于室内深度完成的 RGB 深度融合 GAN)<br>
[paper](https://arxiv.org/abs/2203.10856)<br><br>

[9] Revisiting Domain Generalized Stereo Matching Networks from a Feature Consistency Perspective(从特征一致性的角度重新审视域广义立体匹配网络)<br>
[paper](https://arxiv.org/abs/2203.10887)<br><br>

[8] Deep Depth from Focus with Differential Focus Volume(具有不同焦点体积的焦点深度)<br>
[paper](https://arxiv.org/abs/2112.01712)<br><br>

[7] ChiTransformer:Towards Reliable Stereo from Cues(从线索走向可靠的立体声)<br>
[paper](https://arxiv.org/abs/2203.04554)<br><br>

[6] Rethinking Depth Estimation for Multi-View Stereo: A Unified Representation and Focal Loss(重新思考多视图立体的深度估计：统一表示和焦点损失)<br>
[paper](https://arxiv.org/abs/2201.01501) | [code](https://github.com/prstrive/UniMVSNet)<br><br>

[5] ITSA: An Information-Theoretic Approach to Automatic Shortcut Avoidance and Domain Generalization in Stereo Matching Networks(立体匹配网络中自动避免捷径和域泛化的信息论方法)<br>
keywords: Learning-based Stereo Matching Networks, Single Domain Generalization, Shortcut Learning<br>
[paper](https://arxiv.org/pdf/2201.02263.pdf)<br><br>

[4] Attention Concatenation Volume for Accurate and Efficient Stereo Matching(用于精确和高效立体匹配的注意力连接体积)<br>
keywords: Stereo Matching, cost volume construction, cost aggregation<br>
[paper](https://arxiv.org/pdf/2203.02146.pdf)  | [code](https://github.com/gangweiX/ACVNet)<br><br>

[3] Occlusion-Aware Cost Constructor for Light Field Depth Estimation(光场深度估计的遮挡感知成本构造函数)<br>
[paper](https://arxiv.org/pdf/2203.01576.pdf) | [code](https://github.com/YingqianWang/OACC- Net)<br><br>

[2] NeW CRFs: Neural Window Fully-connected CRFs for Monocular Depth Estimation(用于单目深度估计的神经窗口全连接 CRF)<br>
keywords:  Neural CRFs for Monocular Depth<br>
[paper](https://arxiv.org/pdf/2203.01502.pdf)<br><br>

[1] OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion(通过几何感知融合进行 360 度单目深度估计)<br>
keywords: monocular depth estimation(单目深度估计),transformer<br>
[paper](https://arxiv.org/abs/2203.00838)<br><br>


<br>

<a name="HumanPoseEstimation"/> 

### 人体解析/人体姿态估计(Human Parsing/Human Pose Estimation)

[9] Ray3D: ray-based 3D human pose estimation for monocular absolute 3D localization(用于单目绝对 3D 定位的基于射线的 3D 人体姿态估计)<br>
[paper](https://arxiv.org/abs/2203.11471) | [code](https://github.com/YxZhxn/Ray3D)<br><br>

[8] Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation from Monocular Video(捕捉运动中的人类：来自单目视频的时间注意 3D 人体姿势和形状估计)<br>
[paper](https://arxiv.org/abs/2203.08534) | [video](https://mps-net.github.io/MPS-Net/)<br><br>

[7] Physical Inertial Poser (PIP): Physics-aware Real-time Human Motion Tracking from Sparse Inertial Sensors(来自稀疏惯性传感器的物理感知实时人体运动跟踪)<br>
[paper](https://arxiv.org/abs/2203.08528) | [project](https://xinyu-yi.github.io/PIP/)<br><br>

[6] Distribution-Aware Single-Stage Models for Multi-Person 3D Pose Estimation(用于多人 3D 姿势估计的分布感知单阶段模型)<br>
[paper](https://arxiv.org/abs/2203.07697)<br><br>

[5] MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation(用于 3D 人体姿势估计的多假设transformer)<br>
[paper](https://arxiv.org/abs/2111.12707) | [code](https://github.com/Vegetebird/MHFormer)<br><br>

[4] CDGNet: Class Distribution Guided Network for Human Parsing(用于人类解析的类分布引导网络)<br>
[paper](https://arxiv.org/abs/2111.14173)<br><br>

[3] Forecasting Characteristic 3D Poses of Human Actions(预测人类行为的特征 3D 姿势)<br>
[paper](https://arxiv.org/abs/2011.15079) | [project](https://charposes.christian-diller.de/;) | [video](https://youtu.be/kVhn8OWMgME)<br><br>

[2] Learning Local-Global Contextual Adaptation for Multi-Person Pose Estimation(学习用于多人姿势估计的局部-全局上下文适应)<br>
keywords:Top-Down Pose Estimation(从上至下姿态估计), Limb-based Grouping, Direct Regression<br><br>
[paper](https://arxiv.org/pdf/2109.03622.pdf)<br><br>

[1] MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video(用于视频中 3D 人体姿势估计的 Seq2seq 混合时空编码器)<br>
keywords：3D Human Pose Estimation, Transformer<br>
[paper](https://arxiv.org/pdf/2203.00859.pdf)<br><br>

<br>

<a name="GestureEstimation"/> 

### 手势估计(Gesture Estimation)

[1] ArtiBoost: Boosting Articulated 3D Hand-Object Pose Estimation via Online Exploration and Synthesis(通过在线探索和合成提升关节式 3D 手对象姿势估计)<br>
[paper](https://arxiv.org/abs/2109.05488) | [code](https://github.com/lixiny/ArtiBoost)<br><br>


<br>

<a name="ImageProcessing"/> 


## 图像处理(Image Processing)

<br>

<a name="SuperResolution"/> 

### 超分辨率(Super Resolution)

[10] High-Resolution Image Harmonization via Collaborative Dual Transformations(通过协作双变换实现高分辨率图像协调)<br>
[paper](https://arxiv.org/abs/2109.06671) | [code](https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization)<br><br>

[9] Deep Constrained Least Squares for Blind Image Super-Resolution(用于盲图像超分辨率的深度约束最小二乘)<br>
[paper](https://arxiv.org/abs/2202.07508)<br><br>

[8] Local Texture Estimator for Implicit Representation Function(隐式表示函数的局部纹理估计器)<br>
[paper](https://arxiv.org/abs/2111.08918)<br><br>

[7] A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution(一种用于空间变形鲁棒场景文本图像超分辨率的文本注意网络)<br>
[paper](https://arxiv.org/abs/2203.09388) | [code](https://github.com/mjq11302010044/TATT)<br><br>

[6] Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution(一种真实图像超分辨率的局部判别学习方法)<br>
[paper](https://arxiv.org/abs/2203.09195) | [code](https://github.com/csjliang/LDL)<br><br>

[5] Blind Image Super-resolution with Elaborate Degradation Modeling on Noise and Kernel(对噪声和核进行精细退化建模的盲图像超分辨率)<br>
[paper](https://arxiv.org/abs/2107.00986) | [code](https://github.com/zsyOAOA/BSRDM)<br><br>

[4] Reflash Dropout in Image Super-Resolution(图像超分辨率中的闪退dropout)<br>
[paper](https://arxiv.org/abs/2112.12089)<br><br>

[3] Towards Bidirectional Arbitrary Image Rescaling: Joint Optimization and Cycle Idempotence(迈向双向任意图像缩放：联合优化和循环幂等)<br>
[paper](https://arxiv.org/abs/2203.00911)<br><br>

[2] HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening(用于全色锐化的纹理和光谱特征融合Transformer)<br>
[paper](https://arxiv.org/abs/2203.02503) ｜ [code](https://github.com/wgcban/HyperTransformer)<br><br>

[1] HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging(光谱压缩成像的高分辨率双域学习)<br>
keywords: HSI Reconstruction, Self-Attention Mechanism,  Image Frequency Spectrum Analysis<br>
[paper](https://arxiv.org/pdf/2203.02149.pdf)<br><br>

<br>

<a name="ImageRestoration"/> 

###  图像复原/图像增强/图像重建(Image Restoration/Image Reconstruction)

[5] Exploring and Evaluating Image Restoration Potential in Dynamic Scenes(探索和评估动态场景中的图像复原潜力)<br>
[paper](https://arxiv.org/abs/2203.11754)<br><br>

[4] Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction(通过随机收缩加速逆问题的条件扩散模型)<br>
[paper](https://arxiv.org/abs/2112.05146)<br><br>

[3] Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction(用于高效高光谱图像重建的掩模引导光谱变换器)<br>
[paper](https://arxiv.org/abs/2111.07910) | [code](https://github.com/caiyuanhao1998/MST/)<br><br>

[2] Restormer: Efficient Transformer for High-Resolution Image Restoration(用于高分辨率图像复原的高效transformer)<br>
[paper](https://arxiv.org/abs/2111.09881) | [code](https://github.com/swz30/Restormer)<br><br>

[1] Event-based Video Reconstruction via Potential-assisted Spiking Neural Network(通过电位辅助尖峰神经网络进行基于事件的视频重建)<br>
[paper](https://arxiv.org/pdf/2201.10943.pdf)<br><br>

<br>


<a name="ISR"/> 

### 图像去阴影/去反射(Image Shadow Removal/Image Reflection Removal)

<br>



<a name="ImageDenoising"/> 

### 图像去噪/去模糊/去雨去雾(Image Denoising)

[6] CVF-SID: Cyclic multi-Variate Function for Self-Supervised Image Denoising by Disentangling Noise from Image(通过从图像中分离噪声的自监督图像去噪的循环多变量函数)<br>
[paper](https://arxiv.org/abs/2203.13009) | [code](https://github.com/Reyhanehne/CVF-SID_PyTorch)<br><br>

[5] Unpaired Deep Image Deraining Using Dual Contrastive Learning(使用双重对比学习的非配对深度图像去雨)<br>
[paper](https://arxiv.org/abs/2109.02973) | [code](https://cxtalk.github.io/projects/DCD-GAN.html)<br><br>

[4] AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network(通过非对称 PD 和盲点网络对真实世界图像进行自监督去噪)<br>
[paper](https://arxiv.org/abs/2203.11799) | [code](https://github.com/wooseoklee4/AP-BSN)<br><br>

[3] IDR: Self-Supervised Image Denoising via Iterative Data Refinement(通过迭代数据细化的自监督图像去噪)<br>
[paper](https://arxiv.org/abs/2111.14358) | [code](https://github.com/zhangyi-3/IDR)<br><br>

[2] Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots(具有可见盲点的自监督图像去噪)<br>
[paper](https://arxiv.org/abs/2203.06967) | [code](https://github.com/demonsjin/Blind2Unblind)<br><br>

[1] E-CIR: Event-Enhanced Continuous Intensity Recovery(事件增强的连续强度恢复)<br>
keywords: Event-Enhanced Deblurring, Video Representation<br>
[paper](https://arxiv.org/abs/2203.01935) | [code](https://github.com/chensong1995/E-CIR)<br><br>

<br>

<a name="ImageEdit"/> 

### 图像编辑/图像修复(Image Edit/Inpainting)

[5] High-Fidelity GAN Inversion for Image Attribute Editing(用于图像属性编辑的高保真 GAN 反演)<br>
[paper](https://arxiv.org/abs/2109.06590) | [code](https://github.com/Tengfei-Wang/HFGI) | [project](https://tengfei-wang.github.io/HFGI/)<br><br>

[4] Style Transformer for Image Inversion and Editing(用于图像反转和编辑的样式transformer)<br>
[paper](https://arxiv.org/abs/2203.07932) | [code](https://github.com/sapphire497/style-transformer)<br><br>

[3] MISF: Multi-level Interactive Siamese Filtering for High-Fidelity Image Inpainting(用于高保真图像修复的多级交互式 Siamese 过滤)<br>
[paper](https://arxiv.org/abs/2203.06304) | [code](https://github.com/tsingqguo/misf)<br><br>

[2] HairCLIP: Design Your Hair by Text and Reference Image(通过文本和参考图像设计你的头发)<br>
keywords: Language-Image Pre-Training (CLIP), Generative Adversarial Networks<br>
[paper](https://arxiv.org/abs/2112.05142) | [project](https://github.com/wty-ustc/HairCLIP)<br><br>

[1] Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding(增量transformer结构增强图像修复与掩蔽位置编码)<br>
keywords: Image Inpainting, Transformer, Image Generation<br><br>
[paper](https://arxiv.org/abs/2203.00867) | [code](https://github.com/DQiaole/ZITS_inpainting)<br><br>

<br>

<a name="ImageTranslation"/> 

### 图像翻译(Image Translation)

[5] Maximum Spatial Perturbation Consistency for Unpaired Image-to-Image Translation(未配对图像到图像翻译的最大空间扰动一致性)<br>
[paper](https://arxiv.org/abs/2203.12707) | [code](https://github.com/batmanlab/MSPC)<br><br>

[4] Globetrotter: Connecting Languages by Connecting Images(通过连接图像连接语言)<br>
[paper](https://arxiv.org/abs/2012.04631)<br><br>

[3] QS-Attn: Query-Selected Attention for Contrastive Learning in I2I Translation(图像翻译中对比学习的查询选择注意)<br>
[paper](https://arxiv.org/abs/2203.08483) | [code](https://github.com/sapphire497/query-selected-attention)<br><br>

[2] FlexIT: Towards Flexible Semantic Image Translation(迈向灵活的语义图像翻译)<br>
[paper](https://arxiv.org/abs/2203.04705)<br><br>

[1] Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks(探索图像到图像翻译任务中对比学习的补丁语义关系)<br>
keywords: image translation, knowledge transfer,Contrastive learning<br>
[paper](https://arxiv.org/pdf/2203.01532.pdf)<br><br>

<br>

<a name="IQA"/> 

### 图像质量评估(Image Quality Assessment)

<br>

<a name="StyleTransfer"/> 

### 风格迁移(Style Transfer)

[5] Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer(基于示例的高分辨率肖像风格转移)<br>
[paper](https://arxiv.org/abs/2203.13248) | [code](https://github.com/williamyang1991/DualStyleGAN) | [project](https://www.mmlab-ntu.com/project/dualstylegan/)<br><br>

[4] Industrial Style Transfer with Large-scale Geometric Warping and Content Preservation(具有大规模几何变形和内容保留的工业风格迁移)<br>
[paper](https://arxiv.org/abs/2203.12835) | [project](https://jcyang98.github.io/InST/home.html) | [code](https://github.com/jcyang98/InST)<br><br>

[3] Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization(任意风格迁移和域泛化的精确特征分布匹配)<br>
[paper](https://arxiv.org/abs/2203.07740) | [code](https://github.com/YBZh/EFDM)<br><br>

[2] Style-ERD: Responsive and Coherent Online Motion Style Transfer(响应式和连贯的在线运动风格迁移)<br>
[paper](https://arxiv.org/abs/2203.02574)<br><br>

[1] CLIPstyler: Image Style Transfer with a Single Text Condition(具有单一文本条件的图像风格转移)<br>
keywords: Style Transfer, Text-guided synthesis, Language-Image Pre-Training (CLIP)<br>
[paper](https://arxiv.org/abs/2112.00374)<br><br>


<br>

<a name="Face"/> 

## 人脸(Face)

[5] Cross-Modal Perceptionist: Can Face Geometry be Gleaned from Voices?(跨模态感知者：可以从声音中收集面部几何形状吗？)<br>
[paper](https://arxiv.org/abs/2203.09824) | [project](https://choyingw.github.io/works/Voice2Mesh/index.html)<br><br>

[4] Portrait Eyeglasses and Shadow Removal by Leveraging 3D Synthetic Data(利用 3D 合成数据去除人像眼镜和阴影)<br>
[paper](https://arxiv.org/abs/2203.10474) | [code](https://github.com/StoryMY/take-off-eyeglasses)<br><br>

[3] HP-Capsule: Unsupervised Face Part Discovery by Hierarchical Parsing Capsule Network(分层解析胶囊网络的无监督人脸部分发现)<br>
[paper](https://arxiv.org/abs/2203.10699)<br><br>

[2] FaceFormer: Speech-Driven 3D Facial Animation with Transformers(FaceFormer：带有transformer的语音驱动的 3D 面部动画)<br>
[paper](https://arxiv.org/abs/2112.05329) | [code](https://evelynfan.github.io/audio2face/)<br><br>

[1] Sparse Local Patch Transformer for Robust Face Alignment and Landmarks Inherent Relation Learning(用于鲁棒人脸对齐和地标固有关系学习的稀疏局部补丁transformer)<br>
[paper](https://arxiv.org/abs/2203.06541) | [code](https://github.com/Jiahao-UTS/SLPT-master)<br><br>

<br>

<a name="FacialRecognition"/> 

### 人脸识别/检测(Facial Recognition/Detection)

[4] DeepFace-EMD: Re-ranking Using Patch-wise Earth Mover's Distance Improves Out-Of-Distribution Face Identification(使用 Patch-wise Earth Mover 的距离重新排序改进了分布外人脸识别)<br>
[paper](https://arxiv.org/abs/2112.04016) | [code](https://github.com/anguyen8/deepface-emd)<br><br>

[3] Towards Semi-Supervised Deep Facial Expression Recognition with An Adaptive Confidence Margin(具有自适应置信度的半监督深度面部表情识别)<br>
[paper](https://arxiv.org/abs/2203.12341) | [code](https://github.com/hangyu94/Ada-CM)<br><br>

[2] Privacy-preserving Online AutoML for Domain-Specific Face Detection(用于特定领域人脸检测的隐私保护在线 AutoML)<br>
[paper](https://arxiv.org/abs/2203.08399)<br><br>

[1] An Efficient Training Approach for Very Large Scale Face Recognition(一种有效的超大规模人脸识别训练方法)<br>
[paper](https://arxiv.org/pdf/2105.10375.pdf) | [code](https://github.com/tiandunx/FFC)<br><br>

<br>

<a name="FaceSynthesis"/> 

### 人脸生成/合成/重建/编辑(Face Generation/Face Synthesis/Face Reconstruction/Face Editing)

[3] FENeRF: Face Editing in Neural Radiance Fields(神经辐射场中的人脸编辑)<br>
[paper](https://arxiv.org/abs/2111.15490) | [project](https://mrtornado24.github.io/FENeRF/)<br><br>

[2] GCFSR: a Generative and Controllable Face Super Resolution Method Without Facial and GAN Priors(一种没有面部和 GAN 先验的生成可控人脸超分辨率方法)<br>
[paper](https://arxiv.org/abs/2203.07319)<br><br>

[1] Sparse to Dense Dynamic 3D Facial Expression Generation(稀疏到密集的动态 3D 面部表情生成)<br>
keywords: Facial expression generation, 4D face generation, 3D face modeling<br>
[paper](https://arxiv.org/pdf/2105.07463.pdf)<br><br>

<br>

<a name="FaceAnti-Spoofing"/> 

### 人脸伪造/反欺骗(Face Forgery/Face Anti-Spoofing)

[4] Self-supervised Learning of Adversarial Example: Towards Good Generalizations for Deepfake Detection(对抗样本的自监督学习：迈向 Deepfake 检测的良好泛化)<br>
[paper](https://arxiv.org/abs/2203.12208) | [code](https://github.com/liangchen527/SLADD)<br><br>

[3] Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing(通过 Shuffled Style Assembly 进行域泛化以进行人脸反欺骗)<br>
[paper](https://arxiv.org/abs/2203.05340) | [code](https://github.com/wangzhuo2019/SSAN)<br><br>

[2] Voice-Face Homogeneity Tells Deepfake<br>
[paper](https://arxiv.org/abs/2203.02195) | [code](https://github.com/xaCheng1996/VFD)<br><br>

[1] Protecting Celebrities with Identity Consistency Transformer(使用身份一致性transformer保护名人)<br>
[paper](https://arxiv.org/abs/2203.01318)<br><br>


<br>

<a name="ObjectTracking"/> 

## 目标跟踪(Object Tracking)

[8] Global Tracking Transformers<br>
[paper](https://arxiv.org/abs/2203.13250) | [code](https://github.com/xingyizhou/GTR)<br><br>

[7] Transforming Model Prediction for Tracking(转换模型预测以进行跟踪)<br>
[paper](https://arxiv.org/abs/2203.11192) | [code](https://github.com/visionml/pytracking)<br><br>

[6] MixFormer: End-to-End Tracking with Iterative Mixed Attention(具有迭代混合注意力的端到端跟踪)<br>
[paper](https://arxiv.org/abs/2203.11082) | [code](https://github.com/MCG-NJU/MixFormer)<br><br>

[5] Unsupervised Domain Adaptation for Nighttime Aerial Tracking(夜间空中跟踪的无监督域自适应)<br>
[paper](https://arxiv.org/abs/2203.10541) | [code](https://github.com/vision4robotics/UDAT)<br><br>


[4] Iterative Corresponding Geometry: Fusing Region and Depth for Highly Efficient 3D Tracking of Textureless Objects(迭代对应几何：融合区域和深度以实现无纹理对象的高效 3D 跟踪)<br>
[paper](https://arxiv.org/abs/2203.05334) | [code](https://github.com/DLR- RM/3DObjectTracking)<br><br>

[3] TCTrack: Temporal Contexts for Aerial Tracking(空中跟踪的时间上下文)<br>
[paper](https://arxiv.org/abs/2203.01885) | [code](https://github.com/vision4robotics/TCTrack)<br><br>

[2] Beyond 3D Siamese Tracking: A Motion-Centric Paradigm for 3D Single Object Tracking in Point Clouds(超越 3D 连体跟踪：点云中 3D 单对象跟踪的以运动为中心的范式)<br>
keywords: Single Object Tracking, 3D Multi-object Tracking / Detection, Spatial-temporal Learning on Point Clouds<br>
[paper](https://arxiv.org/abs/2203.01730)<br><br>

[1] Correlation-Aware Deep Tracking(相关感知深度跟踪)<br>
[paper](https://arxiv.org/abs/2203.01666)<br><br>

<br>
<a name="ImageRetrieval"/> 

## 图像&视频检索/视频理解(Image&Video Retrieval/Video Understanding)

[2] Bridging Video-text Retrieval with Multiple Choice Questions(桥接视频文本检索与多项选择题)<br>
[paper](https://arxiv.org/abs/2201.04850) | [code](https://github.com/TencentARC/MCQ)<br><br>

[1] BEVT: BERT Pretraining of Video Transformers(视频Transformer的 BERT 预训练)<br>
keywords: Video understanding, Vision transformers, Self-supervised representation learning, BERT pretraining<br>
[paper](https://arxiv.org/abs/2112.01529) | [code](https://github.com/xyzforever/BEVT)<br><br>



<a name="ActionRecognition"/> 

### 行为识别/动作识别/检测/分割/定位(Action/Activity Recognition)

[14] Unsupervised Pre-training for Temporal Action Localization Tasks(时间动作定位任务的无监督预训练)<br>
[paper](https://arxiv.org/abs/2203.13609) | [code](https://github.com/zhang-can/UP-TAL)<br><br>

[13] Weakly-Supervised Online Action Segmentation in Multi-View Instructional Videos(多视图教学视频中的弱监督在线动作分割)<br>
[paper](https://arxiv.org/abs/2203.13309)<br><br>

[12] How Do You Do It? Fine-Grained Action Understanding with Pseudo-Adverbs(你怎么做呢？ 使用伪副词进行细粒度的动作理解)<br>
[paper](https://arxiv.org/abs/2203.12344)<br><br>

[11] E2(GO)MOTION: Motion Augmented Event Stream for Egocentric Action Recognition(用于以自我为中心的动作识别的运动增强事件流)<br>
[paper](https://arxiv.org/abs/2112.03596)<br><br>

[10] Look for the Change: Learning Object States and State-Modifying Actions from Untrimmed Web Videos(寻找变化：从未修剪的网络视频中学习对象状态和状态修改操作)<br>
[paper](https://arxiv.org/abs/2203.11637) | [code](https://github.com/zju-vipa/MEAT-TIL)<br><br>

[9] DirecFormer: A Directed Attention in Transformer Approach to Robust Action Recognition(鲁棒动作识别的 Transformer 方法中的定向注意)<br>
[paper](https://arxiv.org/abs/2203.10233)<br><br>

[8] Self-supervised Video Transformer(自监督视频transformer)<br>
[paper](https://arxiv.org/abs/2112.01514) | [code](https://git.io/J1juJ)<br><br>

[7] Spatio-temporal Relation Modeling for Few-shot Action Recognition(小样本动作识别的时空关系建模)<br>
[paper](https://arxiv.org/abs/2112.05132) | [code](https://github.com/Anirudh257/strm)<br><br>

[6] RCL: Recurrent Continuous Localization for Temporal Action Detection(用于时间动作检测的循环连续定位)<br>
[paper](https://arxiv.org/abs/2203.07112)<br><br>

[5] OpenTAL: Towards Open Set Temporal Action Localization(走向开放集时间动作定位)<br>
[paper](https://arxiv.org/abs/2203.05114) | [code](https://www.rit.edu/actionlab/opental)<br><br>

[4] End-to-End Semi-Supervised Learning for Video Action Detection(视频动作检测的端到端半监督学习)<br>
[paper](https://arxiv.org/abs/2203.04251)<br><br>

[3] Learnable Irrelevant Modality Dropout for Multimodal Action Recognition on Modality-Specific Annotated Videos(模态特定注释视频上多模态动作识别的可学习不相关模态丢失)<br>
[paper](https://arxiv.org/abs/2203.03014)<br><br>

[2] Weakly Supervised Temporal Action Localization via Representative Snippet Knowledge Propagation(通过代表性片段知识传播的弱监督时间动作定位)<br>
[paper](https://arxiv.org/abs/2203.02925) | [code](https://github.com/LeonHLJ/RSKP)<br><br>

[1] Colar: Effective and Efficient Online Action Detection by Consulting Exemplars(通过咨询示例进行有效且高效的在线动作检测)<br>
keywords:Online action detection(在线动作检测)<br>
[paper](https://arxiv.org/pdf/2203.01057.pdf)<br><br>

<a name="Re-Identification"/> 

### 行人重识别/检测(Re-Identification/Detection)

[1] Cascade Transformers for End-to-End Person Search(用于端到端人员搜索的级联transformer)<br>
[paper](https://arxiv.org/abs/2203.09642) | [code](https://github.com/Kitware/COAT)<br><br>

<a name="VideoCaption"/> 

### 图像/视频字幕(Image/Video Caption)

[3] Open-Domain, Content-based, Multi-modal Fact-checking of Out-of-Context Images via Online Resources(通过在线资源对上下文外图像进行开放域、基于内容、多模式的事实检查)<br>
[paper](https://arxiv.org/abs/2112.00061) | [code](https://s-abdelnabi.github.io/OoC-multi-modal-fc/)<br><br>

[2] Hierarchical Modular Network for Video Captioning(用于视频字幕的分层模块化网络)<br>
[paper](https://arxiv.org/abs/2111.12476) | [code](https://github.com/MarcusNerva/HMN)<br><br>

[1] X -Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning(使用 Transformer 进行 3D 密集字幕的跨模式知识迁移)
keywords：Image Captioning and Dense Captioning(图像字幕/密集字幕)；Knowledge distillation(知识蒸馏)；Transformer；3D Vision(三维视觉)<br>
[paper](https://arxiv.org/pdf/2203.00843.pdf)<br><br>


<a name="MedicalImaging"/> 

## 医学影像(Medical Imaging)

[6] DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning for Histopathology Whole Slide Image Classification(用于组织病理学全幻灯片图像分类的双层特征蒸馏多实例学习)<br>
[paper](https://arxiv.org/abs/2203.12081) | [code](https://github.com/hrzhang1123/DTFD-MIL)<br><br>

[5] ACPL: Anti-curriculum Pseudo-labelling for Semi-supervised Medical Image Classification(半监督医学图像分类的反课程伪标签)<br>
[paper](https://arxiv.org/abs/2111.12918)<br><br>

[4] Vox2Cortex: Fast Explicit Reconstruction of Cortical Surfaces from 3D MRI Scans with Geometric Deep Neural Networks(使用几何深度神经网络从 3D MRI 扫描中快速显式重建皮质表面)<br>
[paper](https://arxiv.org/abs/2203.09446) | [code](https://github.com/ai-med/Vox2Cortex)<br><br>

[3] Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization(通过风格增强和双重归一化的可泛化跨模态医学图像分割)<br>
[paper](https://arxiv.org/abs/2112.11177) | [code](https://github.com/zzzqzhou/Dual-Normalization)<br><br>

[2] Adaptive Early-Learning Correction for Segmentation from Noisy Annotations(从噪声标签中分割的自适应早期学习校正)<br>
keywords: medical-imaging segmentation, Noisy Annotations<br>
[paper](https://arxiv.org/abs/2110.03740) | [code](https://github.com/Kangningthu/ADELE)<br><br>

[1] Temporal Context Matters: Enhancing Single Image Prediction with Disease Progression Representations(时间上下文很重要：使用疾病进展表示增强单图像预测)<br>
keywords: Self-supervised Transformer, Temporal modeling of disease progression<br>
[paper](https://arxiv.org/abs/2203.01933)<br><br>


<a name="TDR"/> 


## 文本检测/识别/理解(Text Detection/Recognition/Understanding)


[3] SwinTextSpotter: Scene Text Spotting via Better Synergy between Text Detection and Text Recognition(通过文本检测和文本识别之间更好的协同作用进行场景文本定位)<br>
[paper](https://arxiv.org/abs/2203.10209) | [code](https://github.com/mxin262/SwinTextSpotter)<br><br>

[2] Fourier Document Restoration for Robust Document Dewarping and Recognition(用于鲁棒文档去扭曲和识别的傅里叶文档恢复)<br>
[paper](https://arxiv.org/abs/2203.09910) | [code](https://sg-vilab.github.io/event/warpdoc/)<br><br>

[1] XYLayoutLM: Towards Layout-Aware Multimodal Networks For Visually-Rich Document Understanding(迈向布局感知多模式网络，以实现视觉丰富的文档理解)<br>
[paper](https://arxiv.org/abs/2203.06947)<br><br>



<a name="RSI"/> 

## 遥感图像(Remote Sensing Image)




<a name="GAN"/> 

## GAN/生成式/对抗式(GAN/Generative/Adversarial)



[14] Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training(了解 Frank-Wolfe 对抗训练并提高效率)<br>
[paper](https://arxiv.org/abs/2012.12368) | [code](https://github.com/TheoT1/FW-AT-Adapt)<br><br>

[13] Feature Statistics Mixing Regularization for Generative Adversarial Networks(生成对抗网络的特征统计混合正则化)<br>
[paper](https://arxiv.org/abs/2112.04120) | [code](https://github.com/naver-ai/FSMR)<br><br>

[12] Subspace Adversarial Training(子空间对抗训练)<br>
[paper](https://arxiv.org/abs/2111.12229) | [code](https://github.com/nblt/Sub-AT)<br><br>

[11] DTA: Physical Camouflage Attacks using Differentiable Transformation Network(使用可微变换网络的物理伪装攻击)<br>
[paper](https://arxiv.org/abs/2203.09831) | [code](https://islab-ai.github.io/dta-cvpr2022/)<br><br>

[10] Improving the Transferability of Targeted Adversarial Examples through Object-Based Diverse Input(通过基于对象的多样化输入提高目标对抗样本的可迁移性)<br>
[paper](https://arxiv.org/abs/2203.09123) | [code](https://github.com/dreamflake/ODI)<br><br>

[9] Towards Practical Certifiable Patch Defense with Vision Transformer(使用 Vision Transformer 实现实用的可认证补丁防御)<br>
[paper](https://arxiv.org/abs/2203.08519)<br>

[8] Few Shot Generative Model Adaption via Relaxed Spatial Structural Alignment(基于松弛空间结构对齐的小样本生成模型自适应)<br>
[paper](https://arxiv.org/abs/2203.04121)<br><br>

[7] Enhancing Adversarial Training with Second-Order Statistics of Weights(使用权重的二阶统计加强对抗训练)<br>
[paper](https://arxiv.org/abs/2203.06020) | [code](https://github.com/Alexkael/S2O)<br><br>

[6] Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack(通过自适应自动攻击对对抗鲁棒性的实际评估)<br>
[paper](https://arxiv.org/abs/2203.05154) | [code1](https://github.com/liuye6666/adaptive_auto_attack) | [code2](https://github.com/liuye6666/adaptive)<br><br>

[5] Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity(对语义相似性的频率驱动的不可察觉的对抗性攻击)<br>
[paper](https://arxiv.org/abs/2203.05151)

[4] Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon(阴影可能很危险：自然现象的隐秘而有效的物理世界对抗性攻击)<br>
[paper](https://arxiv.org/abs/2203.03818)<br><br>

[3] Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer(保护面部隐私：通过风格稳健的化妆转移生成对抗性身份面具)<br>
[paper](https://arxiv.org/pdf/2203.03121.pdf)<br><br>

[2] Adversarial Texture for Fooling Person Detectors in the Physical World(物理世界中愚弄人探测器的对抗性纹理)<br>
[paper](https://arxiv.org/abs/2203.03373)<br><br>

[1] Label-Only Model Inversion Attacks via Boundary Repulsion(通过边界排斥的仅标签模型反转攻击)<br>
[paper](https://arxiv.org/pdf/2203.01925.pdf)<br>


<br>

<a name="IGIS"/> 

## 图像生成/图像合成(Image Generation/Image Synthesis)

[11] Modulated Contrast for Versatile Image Synthesis(用于多功能图像合成的调制对比度)<br>
[paper](https://arxiv.org/abs/2203.09333) | [code](https://github.com/fnzhan/MoNCE)<br><br>

[10] Attribute Group Editing for Reliable Few-shot Image Generation(属性组编辑用于可靠的小样本图像生成)<br>
[paper](https://arxiv.org/abs/2203.08422) | [code](https://github.com/UniBester/AGE)<br><br>

[9] Text to Image Generation with Semantic-Spatial Aware GAN(使用语义空间感知 GAN 生成文本到图像)<br>
[paper](https://arxiv.org/abs/2104.00567) | [code](https://github.com/wtliao/text2image)<br><br>

[8] Playable Environments: Video Manipulation in Space and Time(可播放环境：空间和时间的视频操作)<br>
[paper](https://arxiv.org/abs/2203.01914) | [code](https://willi-menapace.github.io/playable-environments-website)<br><br>


[7] FLAG: Flow-based 3D Avatar Generation from Sparse Observations(从稀疏观察中生成基于流的 3D 头像)<br>
[paper](https://arxiv.org/abs/2203.05789) | [project](https://microsoft.github.io/flag)<br><br>

[6] Dynamic Dual-Output Diffusion Models(动态双输出扩散模型)<br>
[paper](https://arxiv.org/abs/2203.04304)<br><br>

[5] Exploring Dual-task Correlation for Pose Guided Person Image Generation(探索姿势引导人物图像生成的双任务相关性)<br>
[paper](https://arxiv.org/abs/2203.02910) | [code](https://github.com/PangzeCheung/Dual-task-Pose-Transformer-Network)<br><br>



[4] 3D Shape Variational Autoencoder Latent Disentanglement via Mini-Batch Feature Swapping for Bodies and Faces(基于小批量特征交换的三维形状变化自动编码器潜在解纠缠)<br>
[paper](https://arxiv.org/pdf/2111.12448.pdf) | [code](https://github.com/simofoti/3DVAE-SwapDisentangled)<br><br>

[3] Interactive Image Synthesis with Panoptic Layout Generation(具有全景布局生成的交互式图像合成)<br>
[paper])(https://arxiv.org/abs/2203.02104)<br><br>

[2] Polarity Sampling: Quality and Diversity Control of Pre-Trained Generative Networks via Singular Values(极性采样：通过奇异值对预训练生成网络的质量和多样性控制)<br>
[paper](https://arxiv.org/abs/2203.01993) | [demo](http://bit.ly/polarity-demo-colab)<br><br>

[1] Autoregressive Image Generation using Residual Quantization(使用残差量化的自回归图像生成)<br>
[paper](https://arxiv.org/abs/2203.01941) | [code](https://github.com/kakaobrain/rq-vae-transformer)<br><br>





<br>

<a name="3DVision"/> 

## 三维视觉(3D Vision)

[3] The Neurally-Guided Shape Parser: Grammar-based Labeling of 3D Shape Regions with Approximate Inference(神经引导的形状解析器：具有近似推理的 3D 形状区域的基于语法的标记)<br>
[paper](https://arxiv.org/abs/2106.12026) | [code](https://github.com/rkjones4/NGSP)<br><br>

[2] Deep 3D-to-2D Watermarking: Embedding Messages in 3D Meshes and Extracting Them from 2D Renderings(在 3D 网格中嵌入消息并从 2D 渲染中提取它们)<br>
[paper](https://arxiv.org/abs/2104.13450)<br><br>

[1] X -Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning(使用 Transformer 进行 3D 密集字幕的跨模式知识迁移)
关键词：图像字幕/密集字幕；知识蒸馏；Transformer；三维视觉<br>
[paper](https://arxiv.org/pdf/2203.00843.pdf)<br><br>

<br>

<a name="3DPC"/> 

### 点云(Point Cloud)

[12] AziNorm: Exploiting the Radial Symmetry of Point Cloud for Azimuth-Normalized 3D Perception(利用点云的径向对称性进行方位归一化 3D 感知)<br>
[paper](https://arxiv.org/abs/2203.13090) | [code](https://github.com/hustvl/AziNorm)<br><br>

[11] WarpingGAN: Warping Multiple Uniform Priors for Adversarial 3D Point Cloud Generation(为对抗性 3D 点云生成扭曲多个均匀先验)<br>
[paper](https://arxiv.org/abs/2203.12917) | [code](https://github.com/yztang4/WarpingGAN.git)<br><br>

[10] IDEA-Net: Dynamic 3D Point Cloud Interpolation via Deep Embedding Alignment(通过深度嵌入对齐的动态 3D 点云插值)<br>
[paper](https://arxiv.org/abs/2203.11590) | [code](https://github.com/ZENGYIMING-EAMON/IDEA-Net.git)<br><br>

[9] No Pain, Big Gain: Classify Dynamic Point Cloud Sequences with Static Models by Fitting Feature-level Space-time Surfaces(没有痛苦，收获很大：通过拟合特征级时空表面，用静态模型对动态点云序列进行分类)<br>
[paper](https://arxiv.org/abs/2203.11113) | [code](https://github.com/jx-zhong-for-academic-purpose/Kinet)<br><br>

[8] AutoGPart: Intermediate Supervision Search for Generalizable 3D Part Segmentation(通用 3D 零件分割的中间监督搜索)
[paper](https://arxiv.org/abs/2203.06558)<br><br>

[7] Geometric Transformer for Fast and Robust Point Cloud Registration(用于快速和稳健点云配准的几何transformer)<br>
[paper](https://arxiv.org/abs/2202.06688) | [code](https://github.com/qinzheng93/GeoTransformer)<br><br>

[6] Contrastive Boundary Learning for Point Cloud Segmentation(点云分割的对比边界学习)<br>
[paper](https://arxiv.org/abs/2203.05272) | [code](https://github.com/LiyaoTang/contrastBoundary)<br><br>

[5] Shape-invariant 3D Adversarial Point Clouds(形状不变的 3D 对抗点云)<br>
[paper](https://arxiv.org/abs/2203.04041) | [code](https://github.com/shikiw/SI-Adv)<br><br>

[4] ART-Point: Improving Rotation Robustness of Point Cloud Classifiers via Adversarial Rotation(通过对抗旋转提高点云分类器的旋转鲁棒性)<br>
[paper](https://arxiv.org/abs/2203.03888)<br><br>

[3] Lepard: Learning partial point cloud matching in rigid and deformable scenes(Lepard：在刚性和可变形场景中学习部分点云匹配)<br>
[paper](https://arxiv.org/abs/2111.12591) | [code](https://github.com/rabbityl/lepard)<br><br>

[2] A Unified Query-based Paradigm for Point Cloud Understanding(一种基于统一查询的点云理解范式)<br>
[paper](https://arxiv.org/pdf/2203.01252.pdf)<br><br>

[1] CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding(用于 3D 点云理解的自监督跨模态对比学习)<br>
keywords: Self-Supervised Learning, Contrastive Learning, 3D Point Cloud, Representation Learning, Cross-Modal Learning<br>
[paper](https://arxiv.org/abs/2203.00680) | [code](http://github.com/MohamedAfham/CrossPoint)<br><br>

<br>


<a name="3DReconstruction"/> 

### 三维重建(3D Reconstruction)

[11] Neural Reflectance for Shape Recovery with Shadow Handling(使用阴影处理进行形状恢复的神经反射)<br>
[paper](https://arxiv.org/abs/2203.12909) | [code](https://github.com/junxuan-li/Neural-Reflectance-PS)<br><br>

[10] PLAD: Learning to Infer Shape Programs with Pseudo-Labels and Approximate Distributions(学习用伪标签和近似分布推断形状程序)<br>
[paper](https://arxiv.org/abs/2011.13045) | [code](https://github.com/rkjones4/PLAD)<br><br>

[9] ϕ-SfT: Shape-from-Template with a Physics-Based Deformation Model(具有基于物理的变形模型的模板形状)<br>
[paper](https://arxiv.org/abs/2203.11938) | [code](https://4dqv.mpi-inf.mpg.de/phi-SfT/)<br><br>

[8] Input-level Inductive Biases for 3D Reconstruction(用于 3D 重建的输入级归纳偏差)<br>
[paper](https://arxiv.org/abs/2112.03243)<br><br>

[7] AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation(用于 3D 完成、重建和生成的形状先验)<br>
[paper](https://arxiv.org/abs/2203.09516) | [project](https://yccyenchicheng.github.io/AutoSDF/)<br><br>

[6] Interacting Attention Graph for Single Image Two-Hand Reconstruction(单幅图像双手重建的交互注意力图)<br>
[paper](https://arxiv.org/abs/2203.09364) | [code](https://github.com/Dw1010/IntagHand)<br><br>

[5] OcclusionFusion: Occlusion-aware Motion Estimation for Real-time Dynamic 3D Reconstruction(实时动态 3D 重建的遮挡感知运动估计)<br>
[paper](https://arxiv.org/abs/2203.07977) | [project](https://wenbin-lin.github.io/OcclusionFusion)<br><br>


[4] Neural RGB-D Surface Reconstruction(神经 RGB-D 表面重建)<br>
[paper](https://arxiv.org/abs/2104.04532) | [project](https://dazinovic.github.io/neural-rgbd-surface-reconstruction/) | [video](https://youtu.be/iWuSowPsC3g)<br><br>

[3] Neural Face Identification in a 2D Wireframe Projection of a Manifold Object(流形对象的二维线框投影中的神经人脸识别)<br>
[paper](https://arxiv.org/abs/2203.04229) | [code](https://manycore- research.github.io/faceformer) | [project](https://manycore-research.github.io/faceformer)<br><br>

[2] Generating 3D Bio-Printable Patches Using Wound Segmentation and Reconstruction to Treat Diabetic Foot Ulcers(使用伤口分割和重建生成 3D 生物可打印贴片以治疗糖尿病足溃疡)<br>
keywords: semantic segmentation, 3D reconstruction, 3D bio-printers<br>
[paper](https://arxiv.org/pdf/2203.03814.pdf)<br>

[1] H4D: Human 4D Modeling by Learning Neural Compositional Representation(通过学习神经组合表示进行人体 4D 建模)<br>
keywords: 4D Representation(4D 表征),Human Body Estimation(人体姿态估计),Fine-grained Human Reconstruction(细粒度人体重建)<br><br>
[paper](https://arxiv.org/pdf/2203.01247.pdf)<br>

<a name="NeRF"/> 

### 场景重建/视图合成/新视角合成(Novel View Synthesis)

[9] NPBG++: Accelerating Neural Point-Based Graphics(加速基于神经点的图形)<br>
[paper](https://arxiv.org/abs/2203.13318) | [project](https://rakhimovv.github.io/npbgpp/)<br><br>

[8] PlaneMVS: 3D Plane Reconstruction from Multi-View Stereo(从多视图立体重建 3D 平面)<br>
[paper](https://arxiv.org/abs/2203.12082)<br><br>

[7] NeRFusion: Fusing Radiance Fields for Large-Scale Scene Reconstruction(用于大规模场景重建的融合辐射场)<br>
[paper](https://arxiv.org/abs/2203.11283)<br><br>

[6] GeoNeRF: Generalizing NeRF with Geometry Priors(用几何先验概括 NeRF)<br>
[paper](https://arxiv.org/abs/2111.13539) | [code](https://www.idiap.ch/paper/geonerf)<br><br>

[5] StyleMesh: Style Transfer for Indoor 3D Scene Reconstructions(室内 3D 场景重建的风格转换)<br>
[paper](https://arxiv.org/abs/2112.01530) | [code](https://github.com/lukasHoel/stylemesh) | [project](https://lukashoel.github.io/stylemesh/)<br><br>

[4] Look Outside the Room: Synthesizing A Consistent Long-Term 3D Scene Video from A Single Image(向外看：从单个图像合成一致的长期 3D 场景视频)<br>
[paper](https://arxiv.org/abs/2203.09457) | [code](https://github.com/xrenaa/Look-Outside-Room) | [project](https://xrenaa.github.io/look-outside-room/)<br><br>

[3] Point-NeRF: Point-based Neural Radiance Fields(基于点的神经辐射场)<br>
[paper](https://arxiv.org/abs/2201.08845) ｜ [code](https://github.com/Xharlie/pointnerf) |[project](https://xharlie.github.io/projects/project_sites/pointnerf)<br><br>

[2] CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields(文本和图像驱动的神经辐射场操作)<br>
keywords: NeRF,  Image Generation and Manipulation, Language-Image Pre-Training (CLIP)<br>
[paper](https://arxiv.org/abs/2112.05139) | [code](https://cassiepython.github.io/clipnerf/)<br><br>

[1] Point-NeRF: Point-based Neural Radiance Fields(基于点的神经辐射场)<br>
[paper](https://arxiv.org/pdf/2201.08845.pdf) | [code](https://github.com/Xharlie/pointnerf) | [project](https://xharlie.github.io/projects/project_sites/pointnerf/index.html)<br><br>

<a name="ModelCompression"/> 

## 模型压缩(Model Compression)

<br>

<a name="KnowledgeDistillation"/> 

### 知识蒸馏(Knowledge Distillation)

[4] Decoupled Knowledge Distillation(解耦知识蒸馏)<br>
[paper](https://arxiv.org/abs/2203.08679) | [code](https://github.com/megvii-research/mdistiller)<br><br>

[3] Wavelet Knowledge Distillation: Towards Efficient Image-to-Image Translation(小波知识蒸馏：迈向高效的图像到图像转换)<br>
[paper](https://arxiv.org/abs/2203.06321)<br><br>

[2] Knowledge Distillation as Efficient Pre-training: Faster Convergence, Higher Data-efficiency, and Better Transferability(知识蒸馏作为高效的预训练：更快的收敛、更高的数据效率和更好的可迁移性)<br>
[paper](https://arxiv.org/abs/2203.05180) | [code](https://github.com/CVMI-Lab/KDEP)<br><br>

[1] Focal and Global Knowledge Distillation for Detectors(探测器的焦点和全局知识蒸馏)<br>
keywords: Object Detection, Knowledge Distillation<br>
[paper](https://arxiv.org/abs/2111.11837) | [code](https://github.com/yzd-v/FGD)<br><br>

<a name="Pruning"/> 

### 剪枝(Pruning)

[1] Interspace Pruning: Using Adaptive Filter Representations to Improve Training of Sparse CNNs(空间剪枝：使用自适应滤波器表示来改进稀疏 CNN 的训练)<br>
[paper](https://arxiv.org/abs/2203.07808)<br><br>

<a name="Quantization"/> 

### 量化(Quantization)

[2] Implicit Feature Decoupling with Depthwise Quantization(使用深度量化的隐式特征解耦)<br>
[paper](https://arxiv.org/abs/2203.08080)<br><br>

[1] IntraQ: Learning Synthetic Images with Intra-Class Heterogeneity for Zero-Shot Network Quantization(学习具有类内异质性的合成图像以进行零样本网络量化)<br>
[paper](https://arxiv.org/abs/2111.09136) | [code](https://github.com/zysxmu/IntraQ)<br><br>



<br>

<a name="NNS"/> 

## 神经网络结构设计(Neural Network Structure Design)

[2] DyRep: Bootstrapping Training with Dynamic Re-parameterization(使用动态重新参数化的引导训练)<br>
[paper](https://arxiv.org/abs/2203.12868) | [code](https://github.com/hunto/DyRep)<br><br>

[1] BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning(学习探索样本关系以进行鲁棒表征学习)<br>
keywords: sample relationship, data scarcity learning, Contrastive Self-Supervised Learning, long-tailed recognition, zero-shot learning, domain generalization, self-supervised learning<br>
[paper](https://arxiv.org/abs/2203.01522) | [code](https://github.com/zhihou7/BatchFormer)<br><br>

<br>

<a name="CNN"/> 

### CNN

[5] TVConv: Efficient Translation Variant Convolution for Layout-aware Visual Processing(用于布局感知视觉处理的高效翻译变体卷积)(动态卷积)<br>
[paper](https://arxiv.org/abs/2203.10489) | [code](https://github.com/JierunChen/TVConv)<br><br>

[4] On the Integration of Self-Attention and Convolution(自注意力和卷积的整合)<br>
[paper](https://arxiv.org/abs/2111.14556) | [code1](https://github.com/LeapLabTHU/ACmix) | [code2](https://gitee.com/mindspore/models)<br><br>

[3] Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs(将内核扩展到 31x31：重新审视 CNN 中的大型内核设计)<br>
[paper](https://arxiv.org/abs/2203.06717) | [code](https://github.com/megvii-research/RepLKNet)<br>
解读：[凭什么 31x31 大小卷积核的耗时可以和 9x9 卷积差不多？](https://zhuanlan.zhihu.com/p/479182218)<br>
解读：[RepLKNet: 大核卷积+结构重参数让CNN再次伟大](https://zhuanlan.zhihu.com/p/480935774)<br><br>

[2] DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos(视频中稀疏帧差异的端到端 CNN 推断)<br>
keywords: sparse convolutional neural network, video inference accelerating<br>
[paper](https://arxiv.org/abs/2203.03996)<br>

[1] A ConvNet for the 2020s<br>
[paper](https://arxiv.org/abs/2201.03545) | [code](https://github.com/facebookresearch/ConvNeXt)<br>
解读：[“文艺复兴” ConvNet卷土重来，压过Transformer！FAIR重新设计纯卷积新架构](https://mp.weixin.qq.com/s/q-s_dV4-TCiVPMOTZKEgPQ)<br><br>

<br>

<a name="Transformer"/> 

### Transformer

[7] MSG-Transformer: Exchanging Local Spatial Information by Manipulating Messenger Tokens(通过操作信使token交换本地空间信息)<br>
[paper](https://arxiv.org/abs/2105.15168) | [code](https://github.com/hustvl/MSG-Transformer)<br><br>

[6] BoxeR: Box-Attention for 2D and 3D Transformers(用于 2D 和 3D tranformer的 Box-Attention)<br>
[paper](https://arxiv.org/abs/2111.13087) | [code](https://github.com/kienduynguyen/BoxeR)<br><br>

[5] Bootstrapping ViTs: Towards Liberating Vision Transformers from Pre-training(引导 ViT：从预训练中解放视觉transformer)<br>
[paper](https://arxiv.org/abs/2112.03552) | [code](https://github.com/zhfeing/Bootstrapping-ViTs-pytorch)<br><br>

[4] Attribute Surrogates Learning and Spectral Tokens Pooling in Transformers for Few-shot Learning<br>
[paper](https://arxiv.org/abs/2203.09064) | [code](https://github.com/StomachCold/HCTransformers)<br><br>

[3] NomMer: Nominate Synergistic Context in Vision Transformer for Visual Recognition(在视觉transformer中为视觉识别指定协同上下文)<br>
[paper](https://arxiv.org/abs/2111.12994) | [code](https://github.com/TencentYoutuResearch/VisualRecognition-NomMer)<br><br>

[2] Delving Deep into the Generalization of Vision Transformers under Distribution Shifts(深入研究分布变化下的视觉Transformer的泛化)<br>
keywords: out-of-distribution (OOD) generalization, Vision Transformers<br>
[paper](https://arxiv.org/abs/2106.07617) | [code](https://github.com/Phoenix1153/ViT_OOD_generalization)<br><br>

[1] Mobile-Former: Bridging MobileNet and Transformer(连接 MobileNet 和 Transformer)<br>
keywords: Light-weight convolutional neural networks(轻量卷积神经网络),Combination of CNN and ViT<br>
[paper](https://arxiv.org/abs/2108.05895)<br><br>

<br>

<a name="GNN"/> 

### 图神经网络(GNN)


<br>

<a name="NAS"/> 

### 神经网络架构搜索(NAS)

[3] Training-free Transformer Architecture Search(免训练transformer架构搜索)<br>
[paper](https://arxiv.org/abs/2203.12217)<br><br>

[2] Global Convergence of MAML and Theory-Inspired Neural Architecture Search for Few-Shot Learning(MAML 的全局收敛和受理论启发的神经架构搜索以进行 Few-Shot 学习)<br>
[paper](https://arxiv.org/abs/2203.09137) | [code](https://github.com/YiteWang/MetaNTK-NAS)<br><br>

[1] β-DARTS: Beta-Decay Regularization for Differentiable Architecture Search(可微架构搜索的 Beta-Decay 正则化)<br>
[paper](https://arxiv.org/abs/2203.01665)<br><br>

<a name="MLP"/> 

### MLP

[3] Dynamic MLP for Fine-Grained Image Classification by Leveraging Geographical and Temporal Information(利用地理和时间信息进行细粒度图像分类的动态 MLP)<br>
[paper](https://arxiv.org/abs/2203.03253) | [code](https://github.com/ylingfeng/DynamicMLP.git)<br><br>

[2] Revisiting the Transferability of Supervised Pretraining: an MLP Perspective(重新审视监督预训练的可迁移性：MLP 视角)<br>
[paper](https://arxiv.org/abs/2112.00496)<br><br>

[1] An Image Patch is a Wave: Quantum Inspired Vision MLP(图像补丁是波浪：量子启发的视觉 MLP)<br>
[paper](https://arxiv.org/abs/2111.12294) | [code](https://github.com/huawei-noah/CV-Backbones/tree/master/wavemlp_pytorch) | [code](https://gitee.com/mindspore/models/tree/master/research/cv/wave_mlp)<br><br>


<br>

<a name="DataProcessing"/> 

## 数据处理(Data Processing)

[1] Dataset Distillation by Matching Training Trajectories(通过匹配训练轨迹进行数据集蒸馏)(**数据集蒸馏**)<br>
[paper](https://arxiv.org/abs/2203.11932) | [code](https://github.com/GeorgeCazenavette/mtt-distillation) | [project](https://georgecazenavette.github.io/mtt-distillation/)<br><br>

<a name="DataAugmentation"/> 

### 数据增广(Data Augmentation)

[2] TeachAugment: Data Augmentation Optimization Using Teacher Knowledge(使用教师知识进行数据增强优化)<br>
[paper](https://arxiv.org/abs/2202.12513) ｜ [code](https://github.com/DensoITLab/TeachAugment)<br><br>

[1] 3D Common Corruptions and Data Augmentation(3D 常见损坏和数据增强)<br>
keywords: Data Augmentation, Image restoration, Photorealistic image synthesis<br>
[paper](https://arxiv.org/abs/2203.01441) | [projecr](https://3dcommoncorruptions.epfl.ch/)<br><br>



<br>

<a name="BatchNormalization"/> 

### 归一化/正则化(Batch Normalization)

[1] Delving into the Estimation Shift of Batch Normalization in a Network(深入研究网络中批量标准化的估计偏移)<br>
[paper](https://arxiv.org/abs/2203.10778) | [code](https://github.com/huangleiBuaa/XBNBlock)<br><br>

<br>

<a name="ImageClustering"/> 

### 图像聚类(Image Clustering)

[1] RAMA: A Rapid Multicut Algorithm on GPU(GPU 上的快速多切算法)<br>
[paper](https://arxiv.org/abs/2109.01838) | [code](https://github.com/pawelswoboda/RAMA)<br><br>

<br>


<a name="ImageCompression"/> 

### 图像压缩(Image Compression)

[4] Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression(用于高效神经图像压缩的统一多元高斯混合)<br>
[paper](https://arxiv.org/abs/2203.10897) | [code](https://github.com/xiaosu-zhu/McQuic)<br><br>

[3] ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding(具有不均匀分组的空间通道上下文自适应编码的高效学习图像压缩)<br>
[paper](https://arxiv.org/abs/2203.10886)<br><br>

[2] The Devil Is in the Details: Window-based Attention for Image Compression(细节中的魔鬼：图像压缩的基于窗口的注意力)<br>
[paper](https://arxiv.org/abs/2203.08450) | [code](https://github.com/Googolxx/STF)<br><br>

[1] Neural Data-Dependent Transform for Learned Image Compression(用于学习图像压缩的神经数据相关变换)<br>
[paper](https://arxiv.org/abs/2203.04963) | [code](https://dezhao-wang.github.io/Neural-Syntax-Website/) | [project](https://dezhao-wang.github.io/Neural-Syntax-Website/)<br><br>



<br>

<a name="ModelTraining"/> 

## 模型训练/泛化(Model Training/Generalization)

[8] GradViT: Gradient Inversion of Vision Transformers(视觉transformer的梯度反转)<br>
[paper](https://arxiv.org/abs/2203.11894) | [project](https://gradvit.github.io/)<br><br>

[7] Recall@k Surrogate Loss with Large Batches and Similarity Mixup(大批量和相似性混合的 Recall@k 代理损失)<br>
[paper](https://arxiv.org/abs/2108.11179)<br><br>

[6] Out-of-distribution Generalization with Causal Invariant Transformations(具有因果不变变换的分布外泛化)<br>
[paper](https://arxiv.org/abs/2203.11528)<br><br>

[5] Can Neural Nets Learn the Same Model Twice? Investigating Reproducibility and Double Descent from the Decision Boundary Perspective(神经网络可以两次学习相同的模型吗？ 从决策边界的角度研究可重复性和双重下降)<br>
[paper](https://arxiv.org/abs/2203.08124) | [code](https://github.com/somepago/dbViz)<br><br>

[4] Towards Efficient and Scalable Sharpness-Aware Minimization(迈向高效和可扩展的锐度感知最小化)<br>
keywords: Sharp Local Minima, Large-Batch Training<br>
[paper](https://arxiv.org/abs/2203.02714)<br><br>

[3] CAFE: Learning to Condense Dataset by Aligning Features(通过对齐特征学习压缩数据集)<br>
keywords: dataset condensation, coreset selection, generative models<br>
[paper](https://arxiv.org/pdf/2203.01531.pdf) | [code](https://github.com/kaiwang960112/CAFE)<br><br>

[2] The Devil is in the Margin: Margin-based Label Smoothing for Network Calibration(魔鬼在边缘：用于网络校准的基于边缘的标签平滑)<br>
[paper](https://arxiv.org/abs/2111.15430) | [code](https://github.com/by-liu/MbLS)<br><br>

[1] DN-DETR: Accelerate DETR Training by Introducing Query DeNoising(通过引入查询去噪加速 DETR 训练)<br>
keywords: Detection Transformer<br>
[paper](https://arxiv.org/abs/2203.01305) | [code](https://github.com/FengLi-ust/DN-DETR)<br><br>

<br>

<a name="NoisyLabel"/> 

### 噪声标签(Noisy Label)

[2] Scalable Penalized Regression for Noise Detection in Learning with Noisy Labels(带有噪声标签的学习中噪声检测的可扩展惩罚回归)<br>
[paper](https://arxiv.org/abs/2203.07788) | [code](https://github.com/Yikai-Wang/SPR-LNL)<br><br>

[1] Scalable Penalized Regression for Noise Detection in Learning with Noisy Labels(Scalable Penalized Regression for Noise Detection in Learning with Noisy Labels)<br>
[paper](https://arxiv.org/abs/2203.07788) | [code](https://github.com/Yikai-Wang/SPR-LNL)<br><br>



<br>

<a name="Long-Tailed"/> 

### 长尾分布(Long-Tailed Distribution)

[1] Targeted Supervised Contrastive Learning for Long-Tailed Recognition(用于长尾识别的有针对性的监督对比学习)<br>
keywords: Long-Tailed Recognition(长尾识别), Contrastive Learning(对比学习)<br>
[paper](https://arxiv.org/pdf/2111.13998.pdf)<br><br>


<br>

<a name="matching"/> 


## 图像特征提取与匹配(Image feature extraction and matching)

[1] Probabilistic Warp Consistency for Weakly-Supervised Semantic Correspondences(弱监督语义对应的概率扭曲一致性)<br>
[paper](https://arxiv.org/abs/2203.04279) | [code](https://github.com/PruneTruong/DenseMatching)<br><br>

<br>

<a name="VisualRL"/> 

## 视觉表征学习(Visual Representation Learning)

[4] Node Representation Learning in Graph via Node-to-Neighbourhood Mutual Information Maximization(通过节点到邻域互信息最大化的图中节点表示学习)<br>
[paper](https://arxiv.org/abs/2203.12265) | [code](https://github.com/dongwei156/n2n)<br><br>

[3] SimAN: Exploring Self-Supervised Representation Learning of Scene Text via Similarity-Aware Normalization(通过相似性感知归一化探索场景文本的自监督表示学习)<br>
[paper](https://arxiv.org/abs/2203.10492)<br><br>

[2] Exploring Set Similarity for Dense Self-supervised Representation Learning(探索密集自监督表示学习的集合相似性)<br>
[paper](https://arxiv.org/abs/2107.08712)<br><br>

[1] Motion-aware Contrastive Video Representation Learning via Foreground-background Merging(通过前景-背景合并的运动感知对比视频表示学习)<br>
[paper](https://arxiv.org/abs/2109.15130) | [code](https://github.com/Mark12Ding/FAME)<br><br>

<br>

<a name="ModelEvaluation"/> 

## 模型评估(Model Evaluation)



<br>

<a name="MMLearning"/> 

## 多模态学习(Multi-Modal Learning)

[1] MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound(通过视觉、语言和声音的神经脚本知识)<br>
[paper](https://arxiv.org/abs/2201.02639) | [project](https://rowanzellers.com/merlotreserve)<br><br>

<br>

<a name="Audio-VisualLearning"/> 

### 视听学习(Audio-visual Learning)

[3] Self-Supervised Predictive Learning: A Negative-Free Method for Sound Source Localization in Visual Scenes(自监督预测学习：视觉场景中声源定位的无负法方法)(**视觉定位**)<br>
[paper](https://arxiv.org/abs/2203.13412) | [code](https://github.com/zjsong/SSPL)<br><br>

[2] Learning Hierarchical Cross-Modal Association for Co-Speech Gesture Generation(用于协同语音手势生成的学习分层跨模式关联)<br>
[paper](https://arxiv.org/abs/2203.13161) | [project](https://alvinliu0.github.io/projects/HA2G)<br><br>

[1] UMT: Unified Multi-modal Transformers for Joint Video Moment Retrieval and Highlight Detection(用于联合视频时刻检索和高光检测的统一多模态transformer)<br>
[paper](https://arxiv.org/abs/2203.12745) | [code](https://github.com/TencentARC/UMT)<br><br>


<br>

<a name="VLRL"/> 

### 视觉-语言（Vision-language）

[12] LiT: Zero-Shot Transfer with Locked-image text Tuning(带锁定图像文本调整的零样本迁移)<br>
[paper](https://arxiv.org/abs/2111.07991)<br><br>

[11] VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks(视觉和语言任务的参数高效迁移学习)<br>
[paper](https://arxiv.org/abs/2112.06825) | [code](https://github.com/ylsung/VL_adapter)<br><br>

[10] Predict, Prevent, and Evaluate: Disentangled Text-Driven Image Manipulation Empowered by Pre-Trained Vision-Language Model(预测、预防和评估：由预训练的视觉语言模型支持的解耦的文本驱动图像处理)<br>
[paper](https://arxiv.org/abs/2111.13333) | [code](https://github.com/zipengxuc/PPE)<br><br>

[9] LAFITE: Towards Language-Free Training for Text-to-Image Generation(面向文本到图像生成的无语言培训)<br>
[paper](https://arxiv.org/abs/2111.13792) | [code](https://github.com/drboog/Lafite)<br><br>

[8] An Empirical Study of Training End-to-End Vision-and-Language Transformers(培训端到端视觉和语言transformer的实证研究)<br>
[paper](https://arxiv.org/abs/2111.02387) | [code](https://github.com/zdou0830/METER)<br><br>

[7] Pseudo-Q: Generating Pseudo Language Queries for Visual Grounding(为视觉基础生成伪语言查询)<br>
[paper](https://arxiv.org/abs/2203.08481) | [code](https://github.com/LeapLabTHU/Pseudo-Q)<br><br>

[6] Conditional Prompt Learning for Vision-Language Models(视觉语言模型的条件提示学习)<br>
[paper](https://arxiv.org/abs/2203.05557) | [code](https://github.com/KaiyangZhou/CoOp)<br><br>

[5] NLX-GPT: A Model for Natural Language Explanations in Vision and Vision-Language Tasks(视觉和视觉语言任务中的自然语言解释模型)<br>
[paper](https://arxiv.org/abs/2203.05081) | [code](https://github.com/fawazsammani/nlxgpt)<br><br>

[4] L-Verse: Bidirectional Generation Between Image and Text(图像和文本之间的双向生成) **(Oral Presentation)**<br>
[paper](https://arxiv.org/abs/2111.11133)<br><br>

[3] HairCLIP: Design Your Hair by Text and Reference Image(通过文本和参考图像设计你的头发)<br>
keywords: Language-Image Pre-Training (CLIP), Generative Adversarial Networks<br>
[paper](https://arxiv.org/abs/2112.05142) | [project](https://github.com/wty-ustc/HairCLIP)<br><br>

[2] CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields(文本和图像驱动的神经辐射场操作)<br>
keywords: NeRF,  Image Generation and Manipulation, Language-Image Pre-Training (CLIP)<br>
[paper](https://arxiv.org/abs/2112.05139) | [code](https://cassiepython.github.io/clipnerf/)<br><br>

[1] Vision-Language Pre-Training with Triple Contrastive Learning(三重对比学习的视觉语言预训练)<br>
keywords: Vision-language representation learning, Contrastive Learning
[paper](https://arxiv.org/abs/2202.10401) | [code](https://github.com/uta-smile/TCL;)<br><br>


<br>
<a name="Vision-basedPrediction"/> 

## 视觉预测(Vision-based Prediction)

[9] Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion(基于运动不确定性扩散的随机轨迹预测)<br>
[paper](https://arxiv.org/abs/2203.13777) | [code](https://github.com/gutianpei/MID)<br><br>

[8] Non-Probability Sampling Network for Stochastic Human Trajectory Prediction(用于随机人体轨迹预测的非概率采样网络)<br>
[paper](https://arxiv.org/abs/2203.13471) | [code](https://github.com/inhwanbae/NPSN)<br><br>

[7] Remember Intentions: Retrospective-Memory-based Trajectory Prediction(记住意图：基于回顾性记忆的轨迹预测)<br>
[paper](https://arxiv.org/abs/2203.11474) | [code](https://github.com/MediaBrain-SJTU/MemoNet)<br><br>

[6] GaTector: A Unified Framework for Gaze Object Prediction(凝视对象预测的统一框架)<br>
[paper](https://arxiv.org/abs/2112.03549)<br><br>

[5] On Adversarial Robustness of Trajectory Prediction for Autonomous Vehicles(自动驾驶汽车轨迹预测的对抗鲁棒性)<br>
[paper](https://arxiv.org/abs/2201.05057) | [code](https://github.com/zqzqz/AdvTrajectoryPrediction)<br><br>

[4] Adaptive Trajectory Prediction via Transferable GNN(基于可迁移 GNN 的自适应轨迹预测)<br>
[paper](https://arxiv.org/abs/2203.05046)<br><br>

[3] Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective(迈向稳健和自适应运动预测：因果表示视角)<br>
[paper](https://arxiv.org/abs/2111.14820) | [code](https://github.com/vita-epfl/causalmotion)<br><br>

[2] How many Observations are Enough? Knowledge Distillation for Trajectory Forecasting(多少个观察就足够了？ 轨迹预测的知识蒸馏)<br>
keywords: Knowledge Distillation, trajectory forecasting<br>
[paper](https://arxiv.org/abs/2203.04781)<br><br>

[1] Motron: Multimodal Probabilistic Human Motion Forecasting(多模式概率人体运动预测)<br>
[paper](https://arxiv.org/abs/2203.04132)<br><br>



<br>
<a name="Dataset"/> 

## 数据集(Dataset)

[9] Rope3D: TheRoadside Perception Dataset for Autonomous Driving and Monocular 3D Object Detection Task(用于自动驾驶和单目 3D 目标检测任务的路边感知数据集)<br>
[paper](https://arxiv.org/abs/2203.13608) | [dataset](https://thudair.baai.ac.cn/rope)<br><br>

[8] DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation(用于语义变化分割的每日多光谱卫星数据集)<br>
[paper](https://arxiv.org/abs/2203.12560) | [data](https://mediatum.ub.tum.de/1650201) | [website](https://codalab.lisn.upsaclay.fr/competitions/2882)<br><br>

[7] Egocentric Prediction of Action Target in 3D(以自我为中心的 3D 行动目标预测)(**机器人**)<br>
[paper](https://arxiv.org/abs/2203.13116) | [project](https://ai4ce.github.io/EgoPAT3D/)<br><br>

[6] M5Product: Self-harmonized Contrastive Learning for E-commercial Multi-modal Pretraining(电子商务多模态预训练的自协调对比学习)(多模态预训练数据集)<br>
[paper](https://arxiv.org/abs/2109.04275)<br><br>

[5] FERV39k: A Large-Scale Multi-Scene Dataset for Facial Expression Recognition in Videos(用于视频中面部表情识别的大规模多场景数据集)<br>
[paper](https://arxiv.org/abs/2203.09463)<br><br>

[4] Ego4D: Around the World in 3,000 Hours of Egocentric Video(3000 小时以自我为中心的视频环游世界)<br>
[paper](https://arxiv.org/abs/2110.07058) | [project](https://ego4d-data.org/)<br><br>

[3] GrainSpace: A Large-scale Dataset for Fine-grained and Domain-adaptive Recognition of Cereal Grains(用于细粒度和域自适应识别谷物的大规模数据集)<br>
[paper](https://arxiv.org/abs/2203.05306) | [dataset](https://github.com/hellodfan/GrainSpace)<br><br>

[2] Kubric: A scalable dataset generator(Kubric：可扩展的数据集生成器)<br>
[paper](https://arxiv.org/abs/2203.03570) | [code](https://github.com/google-research/kubric)<br><br>

[1] A Large-scale Comprehensive Dataset and Copy-overlap Aware Evaluation Protocol for Segment-level Video Copy Detection(用于分段级视频复制检测的大规模综合数据集和复制重叠感知评估协议)<br>
VCSL (Video Copy Segment Localization) dataset<br>
[paper](https://arxiv.org/abs/2203.02654) | [dataset, metric and benchmark codes](https://github.com/alipay/VCSL)



<br>

<a name="ActiveLearning"/> 

## 主动学习(Active Learning)

[1] Active Learning by Feature Mixing(通过特征混合进行主动学习)<br>
[paper](https://arxiv.org/abs/2203.07034) | [code](https://github.com/Haoqing-Wang/InfoCL)<br><br>



<br>

<a name="Few-shotLearning"/> 

## 小样本学习/零样本学习(Few-shot Learning/Zero-shot Learning)

[3] Ranking Distance Calibration for Cross-Domain Few-Shot Learning(跨域小样本学习的排名距离校准)<br>
[paper](https://arxiv.org/abs/2112.00260)<br><br>

[2] Learning to Affiliate: Mutual Centralized Learning for Few-shot Classification(小样本分类的相互集中学习)<br>
[paper](https://arxiv.org/abs/2106.05517)<br><br>

[1] MSDN: Mutually Semantic Distillation Network for Zero-Shot Learning(用于零样本学习的相互语义蒸馏网络)<br>
keywords: Zero-Shot Learning,  Knowledge Distillation<br>
[paper](https://arxiv.org/abs/2203.03137) | [code](https://github.com/shiming-chen/MSDN)<br><br>



<br>

<a name="ContinualLearning"/> 

## 持续学习(Continual Learning/Life-long Learning)

[4] Probing Representation Forgetting in Supervised and Unsupervised Continual Learning(探索有监督和无监督持续学习中的表征遗忘)<br>
[paper](https://arxiv.org/abs/2203.13381)<br><br>

[3] Meta-attention for ViT-backed Continual Learning(ViT 支持的持续学习的元注意力)<br>
[paper](https://arxiv.org/abs/2203.11684) | [code](https://github.com/zju-vipa/MEAT-TIL)<br><br>

[2] Learning to Prompt for Continual Learning(学习提示持续学习)<br>
[paper](https://arxiv.org/abs/2112.08654) | [code](https://github.com/google-research/l2p)<br><br>

[1] On Generalizing Beyond Domains in Cross-Domain Continual Learning(关于跨域持续学习中的域外泛化)<br>
[paper](https://arxiv.org/abs/2203.03970)<br><br>

<br>

<a name="SG"/> 

## 场景图(Scene Graph)

<a name="SGG"/> 

### 场景图生成(Scene Graph Generation)

[2] Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation(用于无偏场景图生成的堆叠混合注意力和组协作学习)<br>
[paper](https://arxiv.org/abs/2203.09811) | [code](https://github.com/dongxingning/SHA-GCL-for-SGG)<br><br>

[1] Classification-Then-Grounding: Reformulating Video Scene Graphs as Temporal Bipartite Graphs(将视频场景图重新格式化为时间二分图)<br>
keywords: Video Scene Graph Generation, Transformer, Video Grounding<br> 
[paper](https://arxiv.org/abs/2112.04222) | [code](https://github.com/Dawn-LX/VidVRD-tracklets)<br><br>

<br>

<a name="SGP"/> 

### 场景图预测(Scene Graph Prediction)

<br>

<a name="SGU"/> 

### 场景图理解(Scene Graph Understanding)

<br>

<a name="VisualLocalization"/> 

## 视觉定位/位姿估计(Visual Localization/Pose Estimation)

[10] EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation(用于单目物体姿态估计的广义端到端概率透视-n-点)<br>
[paper](https://arxiv.org/abs/2203.13254)<br><br>

[9] RNNPose: Recurrent 6-DoF Object Pose Refinement with Robust Correspondence Field Estimation and Pose Optimization(具有鲁棒对应场估计和姿态优化的递归 6-DoF 对象姿态细化)<br>
[paper](https://arxiv.org/abs/2203.12870) | [code](https://github.com/DecaYale/RNNPose)<br><br>

[8] DiffPoseNet: Direct Differentiable Camera Pose Estimation(直接可微分相机位姿估计)<br>
[paper](https://arxiv.org/abs/2203.11174)<br><br>

[7] ZebraPose: Coarse to Fine Surface Encoding for 6DoF Object Pose Estimation(用于 6DoF 对象姿态估计的粗到细表面编码)<br>
[paper](https://arxiv.org/abs/2203.09418)<br><br>

[6] Object Localization under Single Coarse Point Supervision(单粗点监督下的目标定位)<br>
[paper](https://arxiv.org/abs/2203.09338) | [code](https://github.com/ucas-vg/PointTinyBenchmark/)<br><br>

[5] CrossLoc: Scalable Aerial Localization Assisted by Multimodal Synthetic Data(多模式合成数据辅助的可扩展空中定位)<br>
[paper](https://arxiv.org/abs/2112.09081) | [code](https://github.com/TOPO-EPFL/CrossLoc)<br><br>

[4] GPV-Pose: Category-level Object Pose Estimation via Geometry-guided Point-wise Voting(通过几何引导的逐点投票进行类别级对象位姿估计)<br>
[paper](https://arxiv.org/abs/2203.07918) | [code](https://github.com/lolrudy/GPV_Pose)<br><br>

[3] CPPF: Towards Robust Category-Level 9D Pose Estimation in the Wild(CPPF：在野外实现稳健的类别级 9D 位姿估计)<br>
[paper](https://arxiv.org/abs/2203.03089) | [code](https://github.com/qq456cvb/CPPF)<br><br>

[2] OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation(用于基于深度的 6D 对象位姿估计的对象视点编码)<br>
[paper](https://arxiv.org/abs/2203.01072) | [code](https://github.com/dingdingcai/OVE6D-pose)<br><br>

[1] Spatial Commonsense Graph for Object Localisation in Partial Scenes(局部场景中对象定位的空间常识图)<br>
[paper](https://arxiv.org/abs/2203.05380) | [code](https://github.com/FGiuliari/SpatialCommonsenseGraph-Dataset) | [project](http://fgiuliari.github.io/projects/SpatialCommonsenseGraph/)<br><br>

<br>

<a name="VisualReasoning"/> 

## 视觉推理/视觉问答(Visual Reasoning/VQA)

[2] MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering(基于知识的视觉问答的多模态知识提取与积累)<br>
[paper](https://arxiv.org/abs/2203.09138) | [code](https://github.com/AndersonStra/MuKEA)<br><br>

[1] REX: Reasoning-aware and Grounded Explanation(推理意识和扎根的解释)<br>
[paper](https://arxiv.org/abs/2203.06107) | [code](https://github.com/szzexpoi/rex)<br><br>

<br>

<a name="ImageClassification"/> 

## 图像分类(Image Classification)

[2] CAD: Co-Adapting Discriminative Features for Improved Few-Shot Classification(共同适应判别特征以改进小样本分类)<br>
[paper](https://arxiv.org/abs/2203.13465)<br><br>

[1] GlideNet: Global, Local and Intrinsic based Dense Embedding NETwork for Multi-category Attributes Prediction(用于多类别属性预测的基于全局、局部和内在的密集嵌入网络)<br>
keywords: multi-label classification<br>
[paper](https://arxiv.org/abs/2203.03079) | [code](https://github.com/kareem-metwaly/glidenet) | [project](http://signal.ee.psu.edu/research/glidenet.html)<br><br>

<br>

<a name="domain"/> 

## 迁移学习/domain/自适应(Transfer Learning/Domain Adaptation)

[8] Continual Test-Time Domain Adaptation(持续测试时域适应)<br>
[paper](https://arxiv.org/abs/2203.13591) | [code](https://qin.ee/cotta)<br><br>

[7] Compound Domain Generalization via Meta-Knowledge Encoding(基于元知识编码的复合域泛化)<br>
[paper](https://arxiv.org/abs/2203.13006)<br><br>

[6] Learning Affordance Grounding from Exocentric Images(从离中心图像中学习可供性基础)<br>
[paper](https://arxiv.org/abs/2203.09905) | [code](http://github.com/lhc1224/Cross-View-AG)<br><br>

[5] Category Contrast for Unsupervised Domain Adaptation in Visual Tasks(视觉任务中无监督域适应的类别对比)<br>
[paper](https://arxiv.org/abs/2106.02885)<br><br>

[4] Learning Distinctive Margin toward Active Domain Adaptation(向主动领域适应学习独特的边际)<br>
[paper](https://arxiv.org/abs/2203.05738) | [code](https://github.com/TencentYoutuResearch/ActiveLearning-SDM)<br><br>

[3] How Well Do Sparse Imagenet Models Transfer?(稀疏 Imagenet 模型的迁移效果如何？)<br>
[paper](https://arxiv.org/abs/2111.13445)<br><br>

[2] A Simple Multi-Modality Transfer Learning Baseline for Sign Language Translation(用于手语翻译的简单多模态迁移学习基线)<br>
[paper](https://arxiv.org/abs/2203.04287)<br><br>

[1] Weakly Supervised Object Localization as Domain Adaption(作为域适应的弱监督对象定位)<br>
keywords: Weakly Supervised Object Localization(WSOL), Multi-instance learning based WSOL, Separated-structure based WSOL, Domain Adaption<br>
[paper](https://arxiv.org/abs/2203.01714) | [code](https://github.com/zh460045050/DA-WSOL_CVPR2022)<br><br>

<br>

<a name="MetricLearning"/> 

## 度量学习(Metric Learning)

[4] Hyperbolic Vision Transformers: Combining Improvements in Metric Learning(双曲线视觉transformer：结合度量学习的改进)<br>
[paper](https://arxiv.org/abs/2203.10833) | [code](https://github.com/htdt/hyp_metric)<br><br>

[3] Non-isotropy Regularization for Proxy-based Deep Metric Learning(基于代理的深度度量学习的非各向同性正则化)<br>
[paper](https://arxiv.org/abs/2203.08547) | [code](https://github.com/ExplainableML/NonIsotropicProxyDML)<br><br>

[2] Integrating Language Guidance into Vision-based Deep Metric Learning(将语言指导集成到基于视觉的深度度量学习中)<br>
[paper](https://arxiv.org/abs/2203.08543) | [code](https://github.com/ExplainableML/LanguageGuidance_for_DML)<br><br>

[1] Enhancing Adversarial Robustness for Deep Metric Learning(增强深度度量学习的对抗鲁棒性)<br>
keywords: Adversarial Attack, Adversarial Defense, Deep Metric Learning<br>
[paper](https://arxiv.org/pdf/2203.01439.pdf)<br><br>

<br>

<a name="ContrastiveLearning"/> 

## 对比学习(Contrastive Learning)

[6] Versatile Multi-Modal Pre-Training for Human-Centric Perception(用于以人为中心的感知的多功能多模态预训练)<br>
[paper](https://arxiv.org/abs/2203.13815) | [project](https://hongfz16.github.io/projects/HCMoCo.html;) | [code](https://github.com/hongfz16/HCMoCo)<br><br>

[5] Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation(用于弱监督对象定位和语义分割的类不可知激活图的对比学习)<br>
[paper](https://arxiv.org/abs/2203.13505) | [code](https://github.com/CVI- SZU/CCAM)<br><br>

[4] Rethinking Minimal Sufficient Representation in Contrastive Learning(重新思考对比学习中的最小充分表示)<br>
[paper](https://arxiv.org/abs/2203.07004) | [code](https://github.com/Haoqing-Wang/InfoCL)<br><br>

[3] Selective-Supervised Contrastive Learning with Noisy Labels(带有噪声标签的选择性监督对比学习)<br>
[paper](https://arxiv.org/abs/2203.04181) | [code](https://github.com/ShikunLi/Sel-CL)<br><br>

[2] HCSC: Hierarchical Contrastive Selective Coding(分层对比选择性编码)<br>
keywords: Self-supervised Representation Learning, Deep Clustering, Contrastive Learning<br>
[paper](https://arxiv.org/abs/2202.00455) | [code](https://github.com/gyfastas/HCSC)<br><br>

[1] Crafting Better Contrastive Views for Siamese Representation Learning(为连体表示学习制作更好的对比视图)<br>
[paper](https://arxiv.org/pdf/2202.03278.pdf) | [code](https://github.com/xyupeng/ContrastiveCrop)<br><br>

<br>

<a name="IncrementalLearning"/> 

## 增量学习(Incremental Learning)

[3] Mimicking the Oracle: An Initial Phase Decorrelation Approach for Class Incremental Learning(类增量学习的初始阶段去相关方法)<br>
[paper](https://arxiv.org/abs/2112.04731) | [code](https://github.com/Yujun-Shi/CwD)<br><br>

[2] Forward Compatible Few-Shot Class-Incremental Learning(前后兼容的小样本类增量学习)<br>
[paper](https://arxiv.org/abs/2203.06953) | [code](https://github.com/zhoudw-zdw/CVPR22-Fact)<br><br>

[1] Self-Sustaining Representation Expansion for Non-Exemplar Class-Incremental Learning(非示例类增量学习的自我维持表示扩展)<br>
[paper](https://arxiv.org/abs/2203.06359)<br><br>

<br>

<a name="RL"/> 

## 强化学习(Reinforcement Learning)

[1] Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory(具有编排记忆的演员评论家 GPT 的 3D 舞蹈生成)<br>
[paper](https://arxiv.org/abs/2203.13055) | [code](https://github.com/lisiyao21/Bailando/)<br><br>

<br>

<a name="MetaLearning"/> 

## 元学习(Meta Learning)

[3] A Structured Dictionary Perspective on Implicit Neural Representations(隐式神经表示的结构化字典视角)<br>
[paper](https://arxiv.org/abs/2112.01917) | [code](https://github.com/gortizji/inr_dictionaries)<br><br>

[2] Multidimensional Belief Quantification for Label-Efficient Meta-Learning(标签高效元学习的多维信念量化)<br>
[paper](https://arxiv.org/abs/2203.12768)<br><br>

[1] What Matters For Meta-Learning Vision Regression Tasks?(元学习视觉回归任务的重要性是什么？)<br>
[paper](https://arxiv.org/abs/2203.04905)<br><br>

<br>

<a name="Robotic"/> 

## 机器人(Robotic)

[2] Coarse-to-Fine Q-attention: Efficient Learning for Visual Robotic Manipulation via Discretisation(通过离散化实现视觉机器人操作的高效学习)<br>
[paper](https://arxiv.org/abs/2106.12534) | [code](https://github.com/stepjam/ARM) | [project](https://sites.google.com/view/c2f-q-attention)<br><br>

[1] IFOR: Iterative Flow Minimization for Robotic Object Rearrangement(IFOR：机器人对象重排的迭代流最小化)<br>
[paper](https://arxiv.org/pdf/2202.00732.pdf) | [project](https://imankgoyal.github.io/ifor.html)<br><br>

<br>

<a name="self-supervisedlearning"/> 

## 自监督学习/半监督学习/无监督学习(Self-supervised Learning/Semi-supervised Learning)

[5] SimMatch: Semi-supervised Learning with Similarity Matching(具有相似性匹配的半监督学习)<br>
[paper](https://arxiv.org/abs/2203.06915) | [code](https://github.com/KyleZheng1997/simmatch)<br><br>

[4] Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements(一个完全无监督的框架，用于从噪声和部分测量中学习图像)<br>
[paper](https://arxiv.org/abs/2111.12855) | [code](https://github.com/edongdongchen/REI)<br><br>

[3] UniVIP: A Unified Framework for Self-Supervised Visual Pre-training(自监督视觉预训练的统一框架)<br>
[paper](https://arxiv.org/abs/2203.06965)<br><br>

[2] Class-Aware Contrastive Semi-Supervised Learning(类感知对比半监督学习)<br>
keywords: Semi-Supervised Learning, Self-Supervised Learning, Real-World Unlabeled Data Learning<br>
[paper](https://arxiv.org/abs/2203.02261)<br><br>

[1] A study on the distribution of social biases in self-supervised learning visual models(自监督学习视觉模型中social biases分布的研究)<br>
[paper](https://arxiv.org/pdf/2203.01854.pdf)<br><br>

<br>

<a name="interpretability"/> 

## 神经网络可解释性(Neural Network Interpretability)

[2] Do Explanations Explain? Model Knows Best(解释解释吗？ 模型最清楚)<br>
[paper](https://arxiv.org/abs/2203.02269)<br><br>

[1] Interpretable part-whole hierarchies and conceptual-semantic relationships in neural networks(神经网络中可解释的部分-整体层次结构和概念语义关系)<br>
[paper](https://arxiv.org/abs/2203.03282)<br><br>

<br>

<a name="CrowdCounting"/> 


## 图像计数(Image Counting)

[3] DR.VIC: Decomposition and Reasoning for Video Individual Counting(视频个体计数的分解与推理)<br>
[paper](https://arxiv.org/abs/2203.12335) | [code](https://github.com/taohan10200/DRNet)<br><br>

[2] Represent, Compare, and Learn: A Similarity-Aware Framework for Class-Agnostic Counting(表示、比较和学习：用于类不可知计数的相似性感知框架)<br>
[paper](https://arxiv.org/abs/2203.08354) | [code](https://github.com/flyinglynx/Bilinear-Matching-Network)<br><br>

[1] Boosting Crowd Counting via Multifaceted Attention(通过多方面注意提高人群计数)<br>
[paper](https://arxiv.org/pdf/2203.02636.pdf) | [code](https://github.com/LoraLinH/Boosting-Crowd-Counting-via-Multifaceted-Attention)<br><br>

<br>

<a name="federatedlearning"/> 


## 联邦学习(Federated Learning)

[5] FedCor: Correlation-Based Active Client Selection Strategy for Heterogeneous Federated Learning(用于异构联邦学习的基于相关性的主动客户端选择策略)<br>
[paper](https://arxiv.org/abs/2103.13822)<br><br>

[4] FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction(通过局部漂移解耦和校正与非 IID 数据进行联邦学习)<br>
[paper](https://arxiv.org/abs/2203.11751) | [code](https://github.com/gaoliang13/FedDC)<br><br>

[3] Federated Class-Incremental Learning(联邦类增量学习)<br>
[paper](https://arxiv.org/abs/2203.11473) | [code](https://github.com/conditionWang/FCIL)<br><br>

[2] Fine-tuning Global Model via Data-Free Knowledge Distillation for Non-IID Federated Learning(通过非 IID 联邦学习的无数据知识蒸馏微调全局模型)<br>
[paper](https://arxiv.org/abs/2203.09249)<br><br>

[1] Differentially Private Federated Learning with Local Regularization and Sparsification(局部正则化和稀疏化的差分私有联邦学习)<br>
[paper](https://arxiv.org/abs/2203.03106)<br>

<br><br>

<a name="100"/> 

## 其他

MDAN: Multi-level Dependent Attention Network for Visual Emotion Analysis(用于视觉情感分析的多级依赖注意网络)<br>
[paper](https://arxiv.org/abs/2203.13443)<br><br>

Moving Window Regression: A Novel Approach to Ordinal Regression(序数回归的一种新方法)<br>
[paper](https://arxiv.org/abs/2203.13122) | [code](https://github.com/nhshin-mcl/MWR)<br><br>

Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction(用于有效降维的分层最近邻图嵌入)<br>
[paper](https://arxiv.org/abs/2203.12997) | [code](https://github.com/koulakis/h-nne)<br><br>

TransVPR: Transformer-based place recognition with multi-level attention aggregation(具有多级注意力聚合的基于 Transformer 的**位置识别**)(**图像匹配**)<br>
[paper](https://arxiv.org/abs/2201.02001)<br><br>

Ev-TTA: Test-Time Adaptation for Event-Based Object Recognition(基于事件的对象识别的测试时间适应)<br>
[paper](https://arxiv.org/abs/2203.12247)<br><br>

Learning from All Vehicles(向所有车辆学习)(自动驾驶)<br>
[paper](https://arxiv.org/abs/2203.11934) | [code](https://github.com/dotchen/LAV) | [demo](https://dotchen.github.io/LAV/)<br><br>

Mixed Differential Privacy in Computer Vision(计算机视觉中的混合差分隐私)<br>
[paper](https://arxiv.org/abs/2203.11481)<br><br>

Robust and Accurate Superquadric Recovery: a Probabilistic Approach(稳健且准确的超二次曲线恢复：一种概率方法)<br>
[paper](https://arxiv.org/abs/2111.14517) | [code](http://github.com/bmlklwx/EMS-superquadric_fitting.git)<br><br>

AirObject: A Temporally Evolving Graph Embedding for Object Identification(用于对象识别的时间演化图嵌入)(**object encoding**)<br>
[paper](https://arxiv.org/abs/2111.15150) | [code](https://github.com/Nik-V9/AirObject)<br><br>

FastDOG: Fast Discrete Optimization on GPU(GPU 上的快速离散优化)<br>
[paper](https://arxiv.org/abs/2111.10270) | [code](https://github.com/LPMP/BDD)<br><br>

Neural Collaborative Graph Machines for Table Structure Recognition(用于表结构识别的神经协同图机)<br>
[paper](https://arxiv.org/abs/2111.13359)<br><br>

Contrastive Conditional Neural Processes(对比条件神经过程)<br>
[paper](https://arxiv.org/pdf/2203.03978.pdf)<br><br>

Deep Rectangling for Image Stitching: A Learning Baseline(图像拼接的深度矩形：学习基线)(**Image Stitching**)<br>
[paper](https://arxiv.org/abs/2203.03831) | [code](https://github.com/nie-lang/DeepRectangling)<br><br>

Online Learning of Reusable Abstract Models for Object Goal Navigation(对象目标导航可重用抽象模型的在线学习)<br>
[paper](https://arxiv.org/abs/2203.02583)<br><br>

PINA: Learning a Personalized Implicit Neural Avatar from a Single RGB-D Video Sequence(PINA：从单个 RGB-D 视频序列中学习个性化的隐式神经化身)<br>
[paper](https://arxiv.org/abs/2203.01754) | [video](https://youtu.be/oGpKUuD54Qk) | [project](https://zj-dong.github.io/pina/)<br><br>

<br>

<br>

<a name="2"/> 


# 2. CVPR2022 Oral



[1] L-Verse: Bidirectional Generation Between Image and Text(图像和文本之间的双向生成) **(视觉语言表征学习)**<br>
[paper](https://arxiv.org/abs/2111.11133)<br><br>

<br>

<br>

<a name="3"/> 

# 3. CVPR2022 论文解读汇总

【22】[MLP才是无监督学习比监督学习迁移性能好的关键因素](https://bbs.cvmart.net/articles/6191)<br><br>

【21】[精准高效估计多人3D姿态，美图&北航联合提出分布感知式单阶段模型](https://mp.weixin.qq.com/s/UAtqZezVddSetn6Y_YFq9Q)<br><br>

【20】[利用域自适应思想，北大、字节跳动提出新型弱监督物体定位框架](https://bbs.cvmart.net/articles/6197)<br><br>

【19】[只用一张图+相机走位，AI就能脑补周围环境](https://mp.weixin.qq.com/s/wPiZ5N1bVxFgayej6LbJwQ)<br><br>

【18】[Point-BERT: 基于掩码建模的点云自注意力模型预训练](https://bbs.cvmart.net/articles/6200)<br><br>

【17】[Swin Transformer迎来30亿参数的v2.0，我们应该拥抱视觉大模型吗？](https://bbs.cvmart.net/articles/6202)<br><br>

【16】[Adobe把GAN搞成了缝合怪，凭空P出一张1024分辨率全身人像](https://bbs.cvmart.net/articles/6205)<br><br>

【15】[中国科大等提出点云连续隐式表示 Neural Points：上采样任务效果惊艳](https://bbs.cvmart.net/articles/6201)<br><br>

【14】[马普所开源 ICON：显著提高单张图像重建三维数字人的姿势水平](https://bbs.cvmart.net/articles/6186)<br><br>

【13】[图像也是德布罗意波！华为诺亚&北大提出量子启发 MLP，性能超越 Swin Transfomer](https://bbs.cvmart.net/articles/6182)<br><br>

【12】[群核前沿院等提出首个基于数据驱动的面检测算法](https://bbs.cvmart.net/articles/6184)<br><br>

【11】[MPViT：用于密集预测的多路径视觉Transformer](https://bbs.cvmart.net/articles/6183)<br><br>

【10】[ST++: 半监督语义分割中更优的自训练范式](https://bbs.cvmart.net/articles/6170)<br><br>

【9】[CNN自监督预训练新SOTA！上交等联合提出HCSC：具有层级结构的图像表征自学习新框架](https://mp.weixin.qq.com/s/tVE0Zo0xjKaM4UJ4ouOjhg)<br><br>

【8】[Restormer: 刷新多个low-level任务指标](https://mp.weixin.qq.com/s/tFIZF7sLzJ29jph0_EYyvg)<br><br>

【7】[百变发型！中科大等提出HairCLIP：基于文本和参考图像的头发编辑方法](https://mp.weixin.qq.com/s/v9rExQBXCd3qEbmzb5XKjQ)<br><br>


【6】[凭什么 31x31 大小卷积核的耗时可以和 9x9 卷积差不多？](https://zhuanlan.zhihu.com/p/479182218)
[RepLKNet: 大核卷积+结构重参数让CNN再次伟大](https://zhuanlan.zhihu.com/p/480935774)<br><br>


【5】[U2PL: 使用不可靠伪标签的半监督语义分割](https://bbs.cvmart.net/articles/6163)<br><br>


【4】[针对目标检测的重点与全局知识蒸馏(FGD)](https://bbs.cvmart.net/articles/6169)<br><br>


【3】[即插即用！助力自监督涨点的ContrastiveCrop开源了！](https://bbs.cvmart.net/articles/6157)<br><br>


【2】[从原理和代码详解FAIR的惊艳之作：全新的纯卷积模型ConvNeXt](https://bbs.cvmart.net/articles/6113)
[“文艺复兴” ConvNet卷土重来，压过Transformer！FAIR重新设计纯卷积新架构](https://bbs.cvmart.net/articles/6008)<br><br>


【1】[南开程明明团队和天大提出LD：目标检测的定位蒸馏](https://zhuanlan.zhihu.com/p/474955539)<br><br>



<br>

<a name="4"/> 

# 4. CVPR2022论文分享

<br>

<br>

<a name="5"/> 

# 5. To do list

* CVPR2022 Workshop
