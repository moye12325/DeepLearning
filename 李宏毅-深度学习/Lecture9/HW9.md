# HW9
# 作业九---Explainable AI

# 任务介绍

![](2022-08-07-08-09-03.png)

**本次作业分为两个部分，第一个部分使用了作业三的食物分类的model和dataset，第二个部分使用了作业七的BERT模型相关资料，第一部分要求使用Lime，Saliency Map，Smooth Grad，Filter Visualization，Integrated Gradients5种方法来解释，第二部分则要求使用Attention Visualization，Embedding Visualization，Embedding Analysis这 3种方法**

>本次作业不同于以往的作业，并不会给出一个分数作为评价指标，而会以选择题的形式来检测。作业分成两个部分，分别是作业三和作业七的主题，分别用五种和三种来进行解释。

>但是，我们需要做的选择题的网站（gradescope）是需要课程代码才能进行注册的。

![](2022-08-07-08-26-36.png)

## 作业中的题目

在网上找到了本次作业中的题目

![](2022-08-07-08-37-09.png)

[作业题目链接](http://www.4k8k.xyz/article/weixin_43154149/124641041#Lime_Q14_8)

# 介绍上述方法

如果只是看准确率等评价指标，并不能说明模型的真正好坏，当你加上这些内容的时候会更具有说服力，所以AI的可解释性是非常重要的一个内容。

## 图像CNN系列

### Lime

Local Interpretable Model-agnostic Explanations (LIME)。Lime是一个解释器工具，做的任务主要是对图像以及文本分类任务的解释。LIME的主要思想是利用可解释性模型（如线性模型，决策树）局部近似目标黑盒模型的预测，这个方法不深入模型内部，通过对输入进行轻微的扰动，探测黑盒模型的输出发生何种变化，根据这种变化在兴趣点（原始输入）训练一个可解释性模型。

先把原始图片转成可解释的特征表示，通过可解释的特征表示对样本进行扰动，得到N个扰动后的样本。然后再将这N个样本还原到原始特征空间，并把预测值作为真实值，用可解释的特征数据表示建立简单的数据表示，观察哪些超像素的系数较大。

![](2022-08-07-09-38-51.png)

将这些较大的系数进行可视化可以得到下图的样子，从而理解模型为什么会做出这种判断。青蛙的眼睛和台球很相似，特别是在绿色的背景下。同理红色的心脏也与热气球类似。

![](2022-08-07-09-41-12.png)

LIME通过lime_image.LimeImageExplainer().explain_instance(image,classifier_fn,segmentation_fn)函数输入是要解释的图像，分类器以及语义分割的函数，然后通过线性模型的权重来判断图片的哪一个位置比较重要。再通过get_image_and_mask(label,num_features)函数将需要解释的图像进行一个可视化的表示。

![](2022-08-07-09-32-57.png)

![](2022-08-07-09-33-10.png)


### Saliency Map

思想：将多尺度的特征图结合在一起成为一张地形的显著性图（saliency map），然后利用一个神经网络按照显著性递减的顺序选择关注的位置。该系统能够以一种高效的方式快速选择要进行仔细分析的显著性位置。

Saliency Map方法对图片的每个像素求导，根据导数的绝对值大小来判断像素的重要性。

![](2022-08-07-10-18-26.png)

![](2022-08-07-10-18-36.png)

将对图像影响分类分数最大的像素用高亮显示出来。

>分类分数是神经网络在softmax之前分配给类的输出层中的值，因此它们不是概率，而是通过像softmax这样的函数与概率直接相关。

>saliency map的局限性体现在了人类认知对解释器的确认偏见（Confirmation Bias），就是说不能是这个图迎合了你的认知就能说明这个是分类器把它分为某类的依据。

### Smooth Grad

Smooth Grad是画Saliency Map的一种技术，有的图像在可视化的时候会有很多额外的噪声。可以认为Smooth Grad技术就是为了减少神经网络所看到的一些噪声。它的核心思想就是取一幅感兴趣的图像，然后通过对图像添加噪声对类似图像进行采样，再对每个采样图像的结果灵敏度图取平均值。

Smooth Grad通过在输入图片多次加入随机噪声，对变换后图像求并求均值，达到“引入噪声”来“消除噪声”的效果。

噪声系数在百分20左右时候的效果是最好的

![](2022-08-07-12-56-53.png)

通过增加噪音的方法，使得Saliency Map更加完善，从而更能知道图像中哪些位置是重要的。

![](2022-08-07-13-11-39.png)

![](2022-08-07-13-11-49.png)

### Filter Visualization

当我们想要获得CNN网络中间某一层所观察到的特征的时候，Filter Visualization使用hook函数获得模型中间的层数，并且观察cnn网络中间所观察到的输出内容，以获取他在某一层所学到的东西到底是什么。

~~~python
def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

layer_activations = None
def filter_explanation(x, model, cnnid, filterid, iteration=100, lr=1):
  # x: input image
  # cnnid: cnn layer id
  # filterid: which filter
  model.eval()

  def hook(model, input, output):
    global layer_activations
    layer_activations = output
  
  hook_handle = model.cnn[cnnid].register_forward_hook(hook)
  # When the model forwards through the layer[cnnid], it needs to call the hook function first
  # The hook function save the output of the layer[cnnid]
  # After forwarding, we'll have the loss and the layer activation

  # Filter activation: x passing the filter will generate the activation map
  model(x.cuda()) # forward

  # Based on the filterid given by the function argument, pick up the specific filter's activation map
  # We just need to plot it, so we can detach from graph and save as cpu tensor
  filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
  
  # Filter visualization: find the image that can activate the filter the most
  x = x.cuda()
  x.requires_grad_()
  # input image gradient
  optimizer = Adam([x], lr=lr)
  # Use optimizer to modify the input image to amplify filter activation
  for iter in range(iteration):
    optimizer.zero_grad()
    model(x)
    
    objective = -layer_activations[:, filterid, :, :].sum()
    # We want to maximize the filter activation's summation
    # So we add a negative sign
    
    objective.backward()
    # Calculate the partial differential value of filter activation to input image
    optimizer.step()
    # Modify input image to maximize filter activation
  filter_visualizations = x.detach().cpu().squeeze()

  # Don't forget to remove the hook
  hook_handle.remove()
  # The hook will exist after the model register it, so you have to remove it after used
  # Just register a new hook if you want to use it

  return filter_activations, filter_visualizations
~~~

一般Filter Visualization包括两个类型
* filter activation：观察图片的哪些位置会activate该filter
* filter visualization：什么样的图片能最大程度的activate该filter。

![](2022-08-07-13-47-24.png)

下图是模型第六层第零个filter的activation图

![](2022-08-07-13-47-40.png)

下图是模型第六层第零个filter的visualization图

![](2022-08-07-13-50-01.png)

第23层CNN

![](2022-08-07-14-17-46.png)

### Integrated Gradient

课上讲过一个例子，当动物的鼻子越长，那么这个动物越可能是大象，但是当长度到一个值之后，是大象的可能性却不再回增加。

![](2022-08-07-14-07-55.png)

积分梯度的思想就是既然鼻子太长时**梯度饱和**了，那我就从当前长度开始减短，每减短一点求一次梯度，直到减短到某个称为baseline的最小值（确保在非饱和区，这里设为鼻子长度为0），最后把所有梯度全部加起来。（求和还要乘上一个间隔 △ x i ）

通过使用积分梯度算法对梯度沿不同路径积分。效果图如下，可以看出形状轮廓。

![](2022-08-07-14-19-08.png)

![](2022-08-07-14-19-20.png)

## 第二部分----BERT

### Attention Visualization

在[exbert网站](https://huggingface.co/exbert/)（BERT可视化的工具）中可以输入一个句子，观察在BERT当中，在每一层attention之间的关系。

我们可以双击某个词实现MASK的效果，并且，该工具能够知道我们所覆盖的词是哪一个。

![](2022-08-07-16-46-36.png)

我们也可以通过点击选中某一个头让我们只关注关注某一两个头

![](2022-08-07-16-52-11.png)


### Embedding Visualization

将训练好的BERT模型生成context中单词的embedding，通过PCA降维，就可以对每个单词的2维信息可视化。

选择了第一个阅读理解，查看可视化结果
![](2022-08-07-15-27-27.png)

从第一层可以看出答案离问题离的很近

![](2022-08-07-15-29-19.png)

![](2022-08-07-15-29-52.png)

![](2022-08-07-15-30-13.png)

而到了12层的时候，答案已经远离了其他的单词。


### Embedding Analysis

这一部分的小作业：助教要求我们将euclidean_distance（欧氏距离）和cosine_similarity（余弦距离）的公式写出来。

欧氏距离的计算也就是两点之差的平方和的平方根。

![](2022-08-07-15-47-18.png)

当坐标为数组形式时，可以使用 numpy 模块查找所需的距离。同样，我们也可以使用scipy库的distance.euclidean() 函数返回两点之间的欧几里得距离。
~~~ python
def euclidean_distance(a, b):
    # Compute euclidean distance (L2 norm) between two numpy vectors a and b
    return np.linalg.norm(a-b)
~~~

余弦相似度则是计算两个向量间的夹角的余弦值，计算公式如下：

![](2022-08-07-15-59-28.png)

>余弦距离就是用1减去余弦相似度。
~~~python
def cosine_similarity(a, b):
    # Compute cosine similarity between two numpy vectors a and b
    return 1 - np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
~~~

最后得到结果如下图，上面5个句子描述的是水果苹果，后面5个句子是苹果公司。可以看出，bert能分辨不同语境下‘苹果’的区别。

![](2022-08-07-16-04-13.png)




