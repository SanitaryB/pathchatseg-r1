## 环境问题
### 先安装环境：创建python版本等于3.10的环境conda env create -f environment.yml，理论上wins,linux均兼容
### conda activate demo
### openslide经常报错是正常的，请尝试：
### conda install -c conda-forge openslide

## 环境好了之后，准备运行，全程只需要final.py即可
### 打开final.py
<img width="448" alt="457e63572b85d84ab02b21018d19098" src="https://github.com/user-attachments/assets/6c7a03fd-ab3a-4b77-a0c1-40117b1140db" />


### 指定任意根目录
### 下载配对的tif xml文件到任意目录（百度网盘不断解压即可，xml文件全部都在annotation压缩包里，tif文件需要在center,patient里面再解压得到，但是要注意，tif,xml应该是配对的（即一个患者对应对应标注）
### 输入患者编号加level6，如008_level6,004_node_4_level6，请注意一定要带上level6
### 制定好根目录，tif,xml目录，患者编号后直接运行final.py即可
### 换下一个tif-xml对，运行即可
