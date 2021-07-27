# SegProject

## requirement

    pip install -r requirements.txt

## 配置数据集

请将数据集放在如下位置：

- data
    + train
        - imgs
        - masks
    + val
    + test
 
注意：图像与掩膜的文件名需要一致


## 参数设置

|  参数   | 含义  |
|  :---- : |: ----  :|
| -e  | epochs |
| -b  | batch size |
| -i  | input channels |
| -o  | output channels |
| -l  | learning rate |
| -d  | dataset path |


## 训练

    python train.py -n UNet -e 50 -l 0.03


