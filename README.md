# FairMOT

> [**FairMOT**](http://arxiv.org/abs/2004.01888)，                    
> *IJCV2021 ([arXiv 2004.01888](http://arxiv.org/abs/2004.01888))*

## 摘要
近年来，作为多目标跟踪核心组件的目标检测和重识别（ReID）取得了显著进展。然而，如何在单一网络中完成这两项任务以提高推理速度却鲜受关注。该方向的早期尝试因重识别分支未能有效学习而导致性能下降。本文深入探究了失效的根本原因，并提出了一种解决这些问题的简单基线方法。在30 FPS的实时速度下，该方法显著超越MOT挑战数据集上的现有最优结果。此基线能启发并帮助评估该领域的新思路。



## 跟踪性能

原始论文：

| 数据集   | MOTA | IDF1 | IDS  | MT    | ML    | FPS  |
|----------|------|------|------|-------|-------|------|
|2DMOT15 | 60.6 | 64.7 | 591  | 47.6% | 11.0% | 30.5 |
|MOT16    | 74.9 | 72.8 | 1074 | 44.7% | 15.9% | 25.9 |
|MOT17    | 73.7 | 72.3 | 3303 | 43.2% | 17.3% | 25.9 |
|MOT20    | 61.8 | 67.3 | 5243 | 68.8% | 7.6%  | 13.2 |

![20250709151935](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250709151935.png)

### MOT测试集视频演示

可见github仓库

## 安装
* 克隆本仓库，我们将克隆目录称为${FAIRMOT_ROOT}
* 安装依赖（Python 3.8 和 PyTorch >= 1.7.0）

conda create -n FairMOT
conda activate FairMOT
conda install pytorch1.7.0 torchvision0.8.0 cudatoolkit=10.2 -c pytorch
cd ${FAIRMOT_ROOT}
pip install cython
pip install -r requirements.txt

* 主干网络使用[DCNv2_pytorch_1.7](https://github.com/ifzhang/DCNv2/tree/pytorch_1.7)

git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh



## 数据准备

* **CrowdHuman**
从[官网](https://www.crowdhuman.org)下载数据，目录结构：

```
crowdhuman
   |——————images
   |        └——————train
   |        └——————val
   └——————labels_with_ids
   |         └——————train(empty)
   |         └——————val(empty)
   └------annotation_train.odgt
   └------annotation_val.odgt
```
若需在CrowdHuman上预训练ReID：
```
cd src
python gen_labels_crowd_id.py
```
若需将CrowdHuman加入MIX数据集：
```
cd src
python gen_labels_crowd_det.py
```

* **MIX**
训练数据与[JDE](https://github.com/Zhongdao/Towards-Realtime-MOT)相同，详见其[DATA ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)下载Caltech Pedestrian等数据集。
* **2DMOT15 和 MOT20** 
从[MOT官网](https://motchallenge.net/data/2D_MOT_2015/)下载，目录结构：

```
MOT15
   ——————images
└——————train

        └——————test
   └——————labels_with_ids
            └——————train(空)
MOT20
   ——————images
└——————train

        └——————test
   └——————labels_with_ids
            └——————train(空)
```
生成标签：
```
cd src
python gen_labels_15.py
python gen_labels_20.py
```
2DMOT15的seqinfo.ini文件下载：[[Google]](https://drive.google.com/open?id=1kJYySZy7wyETH4fKMzgJrYUrTfxKlN1w), [[百度，提取码:8o0w]](https://pan.baidu.com/s/1zb5tBW7-YTzWOXpd9IzS0g)

## 预训练模型与基线模型
* **预训练模型**

DLA-34 COCO预训练模型: [DLA-34官方](https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view)
HRNetV2 ImageNet预训练模型: [HRNetV2-W18官方](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw), [HRNetV2-W32官方](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w)
存放路径：
```
${FAIRMOT_ROOT}
   └——————models
           └——————ctdet_coco_dla_2x.pth
           └——————hrnetv2_w32_imagenet_pretrained.pth
           └——————hrnetv2_w18_imagenet_pretrained.pth
```
* **基线模型**

在CrowdHuman上预训练60轮+MIX训练30轮的DLA-34基线模型：
crowdhuman_dla34.pth [[Google]](https://drive.google.com/file/d/1SFOhg_vos_xSYHLMTDGFVZBYjo8cr2fG/view?usp=sharing) [[百度，提取码:ggzx]](https://pan.baidu.com/s/1JZMCVDyQnQCa5veO73YaMw) [[Onedrive]](https://microsoftapc-my.sharepoint.com/:u:/g/personal/v-yifzha_microsoft_com/EUsj0hkTNuhKkj9bo9kE7ZsBpmHvqDz6DylPQPhm94Y08w?e=3OF4XN)
fairmot_dla34.pth (MOT17测试集73.7 MOTA) [[Google]](https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view?usp=sharing) [[百度，提取码:uouv]](https://pan.baidu.com/s/1H1Zp8wrTKDk20_DSPAeEkg) [[Onedrive]](https://microsoftapc-my.sharepoint.com/:u:/g/personal/v-yifzha_microsoft_com/EWHN_RQA08BDoEce_qFW-ogBNUsb0jnxG3pNS3DJ7I8NmQ?e=p0Pul1)
存放路径：
```
${FAIRMOT_ROOT}
   └——————models
           └——————fairmot_dla34.pth
```

## 训练
* 修改src/lib/cfg/data.json中的数据集根目录及src/lib/opts.py中的data_dir
* CrowdHuman预训练+MIX训练：
```bash
sh experiments/crowdhuman_dla34.sh
sh experiments/mix_ft_ch_dla34.sh
```
* 仅MIX训练：
```bash
sh experiments/mix_dla34.sh
```
* 仅MOT17训练：
```bash
sh experiments/mot17_dla34.sh
```
* 基于基线模型微调2DMOT15：
```bash
sh experiments/mot15_ft_mix_dla34.sh
```
* MOT20训练：
需取消注释src/lib/datasets/dataset/jde.py第313-316行：
```python
np.clip(xy[:, 0], 0, width, out=xy[:, 0])
```
...

训练流程：

```bash
sh experiments/crowdhuman_dla34.sh
sh experiments/mix_ft_ch_dla34.sh
sh experiments/mot20_ft_mix_dla34.sh
```

MOT20模型'mot20_fairmot.pth': [[Google]](https://drive.google.com/file/d/1HVzDTrYSSZiVqExqG9rou3zZXX1-GGQn/view?usp=sharing) [[百度，提取码:jmce]](https://pan.baidu.com/s/1bpMtu972ZszsBx4TzIT_CA)
* 消融实验（不同主干网络）：

```
sh experiments/mix_mot17_half_dla34.sh
sh experiments/mix_mot17_half_hrnet18.sh
```


消融模型'mix_mot17_half_dla34.pth': [[Google]](https://drive.google.com/file/d/1dJDGSa6-FMq33XY-cOd_nYxuilv30YDM/view?usp=sharing) [[Onedrive]](https://microsoftapc-my.sharepoint.com/:u:/g/personal/v-yifzha_microsoft_com/ESh1SlUvZudKgUX4A8E3yksBhfRHIf2AsKaaPJ-v_5lVAw?e=NB6UHR) [[百度，提取码:iifa]](https://pan.baidu.com/s/1RQD8ik1labWuwd8jJ-0ukQ)
* 不同训练数据的MOT17测试性能：

| 训练数据        | MOTA | IDF1 | IDS  |
|-----------------|------|------|------|
|MOT17         | 69.8 | 69.9 | 3996 |
|MIX           | 72.9 | 73.2 | 3345 |
|CrowdHuman+MIX| 73.7 | 72.3 | 3303 |

* yolov5s轻量版训练：
```bash
sh experiments/all_yolov5s.sh
```
yolov5s预训练模型: [[Google]](https://drive.google.com/file/d/1Ur3_pa9r3KRY-5qM2cdFhFJ5exghRJvh/view?usp=sharing) [[百度，提取码:wh9h]](https://pan.baidu.com/s/1JHjN_l1nkMnRHRF5TcHYXg)
轻量模型'fairmot_yolov5s': [[Google]](https://drive.google.com/file/d/1MEvsRPyoAqYSCdKaS5Ofrl7ZfKbBZ1Jb/view?usp=sharing) [[百度，提取码:2y3a]](https://pan.baidu.com/s/1dyBEeiGpRfZhqae0c264rg)

## 跟踪
* 2DMOT15验证集跟踪（使用基线模型）：

```bash
cd src
python track.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.6
```
* MOT17消融实验评估：
```bash
cd src
python track_half.py mot --load_model ../exp/mot/mix_mot17_half_dla34.pth --conf_thres 0.4 --val_mot17 True
```
* 生成MOT16/MOT17测试集txt结果：
```bash
cd src
python track.py mot --test_mot17 True --load_model ../models/fairmot_dla34.pth --conf_thres 0.4
```
* 轻量版跟踪（MOT17测试集68.5 MOTA）：
```bash
cd src
python track.py mot --test_mot17 True --load_model ../models/fairmot_yolov5s.pth --conf_thres 0.4 --arch yolo --reid_dim 64
```
结果需提交至[MOT挑战](https://motchallenge.net)评估服务器
* 2DMOT15/MOT20 SOTA结果生成：
```bash
cd src
python track.py mot --test_mot15 True --load_model your_mot15_model.pth --conf_thres 0.3
```

## 演示
输入原始视频生成演示视频：

```bash
cd src
python demo.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.4
```
通过--input-video和--output-root指定输入输出路径，--conf_thres建议0.3~0.7

## 自定义数据集训练
步骤：
1. 每张图像生成一个txt标签文件，每行格式："类别id 中心x/图宽 中心y/图高 宽/图宽 高/图高"
2. 生成包含图像路径的文件（参考src/data/）
3. 在src/lib/cfg/创建数据集JSON配置文件
4. 训练时添加参数：`--data_cfg '../src/lib/cfg/your_dataset.json'`

## 致谢
代码参考[Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT)和[xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)

