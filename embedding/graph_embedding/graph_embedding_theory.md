[TOC]



# 知识

- word2vec目标

> 将稀疏的词语one-hot向量表达位稠密的embedding向量，使得语义相近的词语在embedding空间中距离更近。为了解决这个问题，我们假设语料库中，上下文相邻的词语之间的语义互相依赖，因此语料库中的每一个词语以及它的上下文之前的其它词语之间可以互相推测。

- graph-embedding 的主要作用

> 不仅限于时间序列化的数据，而且广泛应用在图结构的语料数据。
>
> 传统的方法会忽略用户行为的时间因素，而盲目的认为某个用户相关的所有商品都有着相近的语义，事实上我们知道用户的兴趣是会随着时间变化的，这篇文章中，作者仅仅选取用户兴趣的一个时间窗。


# word2vec

## CBOW

- 目的

> 将多个语境单词放在输入层，然后共同预测输出层的一个单词目标服务。
>

## Skip-Gram

- 目的

> 将一个单词放在输入层，为预测输出层的多个语境单词服务。


[word2vec Parameter Learning Explained读书笔记](https://zhuanlan.zhihu.com/p/64430221)

# reference

[c++图的基本操作_阿里巴巴图模型的商品推荐embedding计算](https://blog.csdn.net/weixin_34043312/article/details/113050197)

[TODO Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba. 2018]()

[TODO DeepWalk -基于图的embedding向量 - 读书笔记](https://zhuanlan.zhihu.com/p/67816202)
