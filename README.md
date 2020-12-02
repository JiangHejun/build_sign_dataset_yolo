# 本项目主要用于为yolov5-pytorch生成签名识别数据
* [x] 支持多签名识别
* [x] 图像缩放
* [x] 图像加噪：椒盐、高斯
* [x] 图像选装
* [x] 签名抠图
* **在运行`python3 build.py`之前，请先运行`python3 build.py --finetune`进行微调**
* **[background](./dataset/background)和[signature](./dataset/signature)的目录结构固定，不可改变；增加新的标签（签名）请在[signature](./dataset/signature)目录之下新建对应的文件夹**