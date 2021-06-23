[[TOC]]



# [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)

- 论文指出，目前CTR预估中，存在两个问题：

输入的数据，比如用户数据和商品数据，这些都是高纬度和稀疏的特征，容易造成过拟合；

有效的预估中，需要进行特征交叉。这其中又会涉及到大量的人工标记评估时间

因此本论文提出了新的模型AutoInt用来学习高阶的输入特征，用来解决稀疏且高纬度的输入特征。 同时这个模型能够同时处理数字型（numerical）和分类型（categorical）的特征。

- 论文的贡献

论文的模型能够进行显示学习高阶特征，同时能够找到很好的解释方法（其实也就是用了attention机制来解释）

提出了一个基于self-attention神经网络，它能够自动学习高阶特征，同时有效解决高纬度的稀疏数据问题

实验中展示出论文中提出模型达到了SOTA，且有更好的可解释性；

# [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba]()

- 论文贡献

以前的工作采用的是embedding和MLP（普通的全量连接层），原始的特征会直接映射成低纬度vectors，然后利用MLP作为最后的预测层。但是以前的这种方法没有考虑到用户的行为序列，仅仅是把原有的特征进行拼接。 本论文的贡献是：

利用Transformer模型来捕捉底层用户行为的信号；

实验结果证明新提出的模型在CTR预估上有重大的提升；

# [Deep Multifaceted Transformers for Multi-objective Ranking in Large-Scale E-commerce Recommender Systems]()

- 论文贡献

以前的CTR模型仅仅集中在历史的点击序列中，但没有关注用户的多种行为。因此本论文提出了模型Deep Multifaceted Transformers (DMT)来考虑用户的多种行为，并应用多任务学习方法来训练模型。论文的主要贡献如下：

多任务学习方法：同时使用CTR（点击率）和Conversion Rate （CVR，转化率）来训练模型

引入用户的多种不同行为：用户有很多不同的行为，比如：点击、添加商品到购物车，排序。

引入偏差隐藏反馈（bias）：

（1）位置偏差：一个用户会点击一个商品仅仅是因为它排名比较高，这种现象称为“位置偏差”

（2）邻居偏差：用户偏向于点击目标商品的附近商品

# reference

[transformers在ctr中的应用](https://zhuanlan.zhihu.com/p/349509932)

[AutoInt github实现](https://github.com/shichence/AutoInt)

[AutoInt 论文中代码实现](https://github.com/DeepGraphLearning/RecommenderSystems)
