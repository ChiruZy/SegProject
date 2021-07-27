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

Glas 数据集下载：https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest

## 参数设置


<table>
<tr>
<th align=center>参数</th>
<th align=center>默认</th>
<th align=center>含义</th>
</tr>
<tr>
<th align=center>-e</th>
<th align=center>50</th>
<th align=center>epochs</th>
</tr>
<tr>
<th align=center>-b</th>
<th align=center>4</th>
<th align=center>batch size</th>
</tr>
<tr>
<th align=center>-i</th>
<th align=center>3</th>
<th align=center>input channels</th>
</tr>
<tr>
<th align=center>-o</th>
<th align=center>1</th>
<th align=center>output channels</th>
</tr>
<tr>
<th align=center>-l</th>
<th align=center>0.03</th>
<th align=center>learning rate</th>
</tr>
<tr>
<th align=center>-b</th>
<th align=center>datas/Glas</th>
<th align=center>dataset path</th>
</tr>
</table>


## 训练

    python train.py -n UNet -e 50 -l 0.03


