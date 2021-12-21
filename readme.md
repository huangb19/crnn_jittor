Convolutional Recurrent Neural Network
======================================

本仓库为Convolutional Recurrent Neural Network (CRNN) 的[Jittor](https://cg.cs.tsinghua.edu.cn/jittor/)实现版本

模型和训练部分基于[Pytorch版本](https://github.com/meijieru/crnn.pytorch)修改，[原论文仓库](https://github.com/bgshih/crnn)为Lua实现




运行demo程序
--------
```bash
python3 demo.py 
```

demo程序读取示例图片并且进行识别



示例图片1：![demo-1](images/demo.png)

期望输出：

```
using cuda
loading pretrained model from ./expr/crnn.pkl
['a-----vv-a-i-l-a--b-l-ee--'] => ['available']       
```



示例图片2：![demo-2](images/demo2.jpg)

期望输出：

```
using cuda
loading pretrained model from ./expr/crnn.pkl
['s--h-a--k-e--s-h--a-c--k--'] => ['shakeshack']   
```



训练你自己的模型
-----------------

使用[MJSynth](https://thor.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz)数据集进行训练，其包含人工合成的8.9M张图像。解压数据为图片格式，文件的IO容易成为训练速度的瓶颈，我们提供了两种方式用于读取数据集并且训练。



* 原始数据读取：操作简单，但速度慢

  * 将`train.py`中52至62行替换为64至76行，并且设置64行的`dataset_root`为数据集目录，运行`python3 train.py`
* lmdb数据集读取：制作完成后IO速度加快数十倍，但制作该数据集耗时较长。
  * 设置`create_dataset.py`92行的`dataset_root`为数据集目录，并且运行（可能需要运行数天）
  * 运行`python3 train.py --trainRoot="./dataset/train" --valRoot="./dataset/val"`  



更多的参数可以通过`python train.py --help`查看，训练参考：

在1个epoch后验证集准确率应达到80%，5个epoch后达到95%，25个epoch后达到98%左右



## 测试

依照原论文实验设定，需要测试CRNN在各个数据集上无词表和有词表转录的词准确率。词表是一组备选单词，图片对应的单词应该在词表中。例如对于图片：![demo-1](./images/demo.png)

对应的一个大小为4的词表可能为：

```
Artificial
Available
Availabil
Intelligence
```

假设模型识别出的序列为$\pi$，我们需要找到词表中所有与$\pi$的编辑距离(Levenshtein Distance)不超过`threshold=3`的单词，并且依次以这些单词为Target计算CTCLoss，取Loss最小的单词为识别结果。

测试集为真实世界取景的`IC03`,`IC13`,`IIIT5K`,`SVT`等，其中`SVT`数据集自带词表，其它数据集需要自己构建。在测试`IC03`数据集时，我们还采用了50k-words Hunspell词典作为词表，由于该词表很大且固定，寻找所有编辑距离不超过`threshold`的单词需要使用特定数据结构`BK-Tree`加速。

测试集应该处理成统一格式便于输入，每个数据集文件夹下应有以下文件：

```
|----test_dataset
   |----SVT
       |----gt.txt   每一行为图片相对路径和其标注
       |----(lex.txt) （仅SVT需要）
       |----images
          |--....（所有图片）
```

你可以自己处理数据集，也可以以下命令获取已经处理好的数据集和词表：

````bash
git submodule init
git submodule update
````

运行`python test.py`即可得到测试结果



## 更多的架构

我们还提供了原论文之外的许多模型架构用于该任务，并且增加了训练过程记录等功能。它们的文件结构和功能分别如下：

- record/ 训练过程的数据记录
- model_comb.py 支持模块替换和自定义扩展的模型
- train_comb.py 与可替换、可扩展模型对应的训练方法
- result.py 用于查看record/下的记录，作用是打印出每个epoch的验证集数据中最高的准确率及其对应的loss



具体来说，你可以查看`train_comb.py`的第86行及之后的部分，用于选取模块的组合方式。其它的训练过程与之前完全相同。



如果你需要自己对模型进行扩展，请参考：

- 你需要查看`model_comb.py`文件，该文件包含如下部分
  - 一些可能被反复使用的子组件
  - 直接对应于“特征提取-特征优化-序列生成”模型范式某个部分的结构，例如对应于特征提取部分的残差网络结构
  - 顶层模块，也就是完整的模型
- 如果想要实现自己的替换组件，请这样做：
  - 在`model_comb.py`的第二部分定义自己的模块
    - 你需要确保输入或输出的张量形状与已有模块保持一致，可以参考已有模块进行形状调整
    - 模型范式中的"特征优化"部分，在代码中包含在了RNN模块中，也就是self.reoptimize层，你可以参考已有组件进行定义
    - RNN模块需要额外地定义损失函数(因为不同结构所用的损失函数可能不同，为了保证模块自由替换我们需要使用这种写法)，定义位于calc_loss()函数中
    - RNN模块的self.output_dense变量用于指导validate过程，这是因为我们采用了兼容CTC规则的validate方法；简单来讲，如果你用的是CTC规则，就把这个变量设置为False，否则设置为True
  - 在第三部分顶层模块中，引用自己定义的模块
  - 在`train_comb.py`中定义使用新定义模块的组合方式，并修改test_set变量以使用该组合方式
