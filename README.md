# ICDAR2019广告牌文本检测  


一、论文部分  
论文名称：《Detecting Text in Natural Image with Connectionist Text Proposal Network》  
论文链接：  https://arxiv.org/pdf/1609.03605.pdf  
tensorflow代码： https://github.com/eragonruan/text-detection-ctpn   

数据集：自然场景文本数据集(ReCTS)，该数据集包含25，000幅图像，由大量的招牌组成。在数据集中，所有文本行和字符都被标记为位置和字符代码。  

1. CTPN是在ECCV 2016提出的一种文字检测算法。CTPN结合CNN与LSTM深度网络，能有效的检测出复杂场景的横向分布的文字，效果如图1。  
                   ![1](https://github.com/JingJLiu/ICDAR2019.github.io/blob/master/train_ReCTS_001867.jpg)   
                                                       图1 场景文本检测（图像出自ICDAR2019测试数据）  
                                    
2. CPTN网络结构  
    原始CTPN只检测横向排列的文字。CTPN结构与Faster R-CNN基本类似，但是加入了LSTM层。网络结构如图：  
                ![2](https://github.com/JingJLiu/ICDAR2019.github.io/blob/master/v2-b29f366f73ac0fba695435770e85809e_r.jpg)  
                                                                  图2 CPTN网络结构  
(1) 假设输入 N Image，利用VGG16提取特征，得到conv5_3的特征作为feature map，大小为N×C×H×W，feature map的宽高都是Image的1/16；   
        （VGG16介绍：输入图像224×224×3，则conv5_3的featur map为14×14×512。  https://www.sohu.com/a/241338315_787107）  
(2) 在conv5_3feature map上做3×3的滑动窗口，如下图，即每个点都结合周围3×3区域特征获得一个长度为3×3×C的特征向量。输出N×9C×H×W的feature map，该特征显然只有CNN学习到的空间特征。再将这个feature map进行Reshape成（NH）×W×9C。  
             ![1](https://github.com/JingJLiu/ICDAR2019.github.io/blob/master/v2-4399a8ecb012241fa542e084eb7d727f_r.jpg)  
然后以Batch=NH,最大时间长度 Tmax=W 的数据流输入BLSTM，学习每一行的序列特征。BLSTM输出为（NH）×W×256，再Reshape恢复为N×256×H×W，该特征既包含空间特征，也包含了LSTM学习到的序列特征。  
    
在具体代码中，3×3的滑块用3×3的卷积代替  (../nets/model_train）  

    def model(image):
        image = mean_image_subtraction(image)
        with slim.arg_scope(vgg.vgg_arg_scope()):
            conv5_3 = vgg.vgg_16(image)
        rpn_conv = slim.conv2d(conv5_3, 512, 3)  
        
 (3) 经过FC层，变为 N×512×H×W 的特征，最后经过类似Faster R-CNN的RPN网络，获得text proposals，如图2（b）。  
 
 二、实验部分  
 1. 环境配置：  
 ![1](https://github.com/JingJLiu/ICDAR2019.github.io/blob/master/train_ReCTS_001867.jpg)  
 
 2.编译源码：  
 
     cd lib/utils
     chmod +x Make.sh | Design and Development
     ./make.sh  
 3.准备数据：
   比赛原始数据：./ICDAR2019/ReCTS/ReCTS_part3/   将ICDAR2019的.xml文件处理成.txt且，标注数据时为顺时针方向。坐标（x1,y1,x2,y2,x3,y3,x4,y4）依次表示左上、右上、左下、右下。  
  ![1](https://github.com/JingJLiu/ICDAR2019.github.io/blob/master/train_ReCTS_001867.jpg) 
   
 

