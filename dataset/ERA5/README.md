## 我们自己准备的数据集，从EC再分析数据中选择一个子集

## 原始数据下载
从这里下载：
https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset

搜索：
![image](https://github.com/MAZiqing/PointFormer/assets/44238026/30436b5e-9850-4689-afa7-62dc28e2cb27)

选择一些气象要素，注意要和图里选的一样；选择一样的经纬度区域
![image](https://github.com/MAZiqing/PointFormer/assets/44238026/803a01d7-7731-4158-af75-971d3dcffcb9)

![image](https://github.com/MAZiqing/PointFormer/assets/44238026/a9a671d3-35bc-4171-a0cf-9d2901aceb03)

![image](https://github.com/MAZiqing/PointFormer/assets/44238026/e8ce3b7d-2d13-4a98-b073-4b4f4ca7bff0)

我是每年选择3个月，下载一次，大小500MB。重复此操作，直到下载完我们所需的全部3年（或10年）的数据。

## 数据处理
数据放在 ./raw/china_raw/ 目录下面，该目录下面可以有多个文件，命名需要可以按时间sort就行
![image](https://github.com/MAZiqing/PointFormer/assets/44238026/02c16f29-4071-428e-9619-7994e0f26353)



