[TOC]


# 推荐系统

## 推荐系统的目的

### 发掘长尾

理论基础是长尾理论，一小部分最热的资源得到大量的关注，而剩下的一大部分资源却无人问津，这不仅导致了资源的浪费，也让小众的用户无法找到自己感兴趣的内容

### 解决信息过载

提高信息利用率

### 提高点击率/转化率

### 个性化服务

# 召回

## 召回模块的演进过程 (TODO)

### 布尔召回 - 基于倒排索引

https://www.jianshu.com/p/86bd98551f4b

### 向量化召回

#### 向量化召回

主要通过模型来学习用户和物品的兴趣向量，通过内积来计算用户和物品之间的相似性。比较经典的模型有：

- YouTube DNN将用户和item的特征转成向量进行相似度检索

- 基于深度语义检索模型（DSSM）

输入：用户query、与该query相关的项目D1，两个不相关的D2和D3。将query、D1、D2、D3都转成embedding，进入NN神经网络。

输出：最后softmax给三种项目打分α，正例分数尽可能高。

#### 向量化检索

但是在实际线上应用时，由于物品空间巨大，计算用户兴趣向量和所有物品兴趣向量的内积，耗时十分巨大，有时候会通过局部敏感Hash等方法来进行近似求解。

##### 向量检索索引

- 基于树的方法

KD树是其下的经典算法。一般而言，在空间维度比较低时，KD树的查找性能还是比较高效的；但当空间维度较高时，该方法会退化为暴力枚举，性能较差，这时一般会采用下面的哈希方法或者矢量量化方法。因为高维空间的主要特点之一是，随着维数增加，任意两点之间最大距离与最小距离趋于相等

- 哈希方法

LSH(Locality-Sensitive Hashing)是其下的代表算法。文献[7]是一篇非常好的LSH入门资料。 对于小数据集和中规模的数据集(几个million-几十个million)，基于LSH的方法的效果和性能都还不错。这方面有2个开源工具FALCONN和NMSLIB。

LSH：https://blog.csdn.net/leadai/article/details/89391366

LSH的基本思想如下：我们首先对原始数据空间中的向量进行hash映射，得到一个hash table，我们希望，原始数据空间中的两个相邻向量通过相同的hash变换后，被映射到同一个桶的概率很大，而不相邻的向量被映射到同一个桶的概率很小。因此，在召回阶段，我们便可以将所有的物品兴趣向量映射到不同的桶内，然后将用户兴趣向量映射到桶内，此时，只需要将用户向量跟该桶内的物品向量求内积即可。这样计算量被大大减小。

关键的问题是，如何确定hash-function？在LSH中，合适的hash-function需要满足下面两个条件：
1）如果d(x,y) ≤ d1， 则h(x) = h(y)的概率至少为p1；
2）如果d(x,y) ≥ d2， 则h(x) = h(y)的概率至多为p2；
其中d(x,y)表示x和y之间的距离， h(x)和h(y)分别表示对x和y进行hash变换。

满足以上两个条件的hash function称为(d1,d2,p1,p2)-sensitive。而通过一个或多个(d1,d2,p1,p2)-sensitive的hash function对原始数据集合进行hashing生成一个或多个hash table的过程称为Locality-sensitive Hashing。



- 矢量量化方法

矢量量化方法，即vector quantization。在矢量量化编码中，关键是码本的建立和码字搜索算法。比如常见的聚类算法，就是一种矢量量化方法。而在相似搜索中，向量量化方法又以PQ方法最为典型。
对于大规模数据集(几百个million以上)，基于矢量量化的方法是一个明智的选择，可以用用Faiss开源工具。

https://blog.csdn.net/gaoyanjie55/article/details/81383434 （机器学习-向量检索+存储格式技术）


### 深度树索引

布尔召回：倒排表的key-value的value值会越来越长，检索性能会越来越差。

向量检索：局限于模型的向量表达是否够好，局限于向量空间。

基于深度树的检索：解决检索性能以及全量检索。

【全量检索：因为协同过滤在召回的时候，并不能真正的面向全量商品库来做检索，系统只能在用户历史行为过的商品里面找到侯选的相似商品来做召回】

[AI在视频广告中的应用（重点看深度数召回）](https://www.cnblogs.com/Lee-yl/p/11333535.html)

[向量检索在闲鱼视频去重的实践](https://juejin.im/entry/5b91f8075188255c80664191)


# reference

[针对2.2亿人的推荐系统的重构技术解析](https://dbaplus.cn/news-141-2482-1.html)


[基于Embedding的推荐系统召回策略(实战篇)](https://www.csuldw.com/2019/02/06/2019-02-06-recommendation-with-neural-network-embeddings/)

[推荐场景中召回模型的演化过程](https://cloud.tencent.com/developer/news/564779)


[召回和排序常用算法实现](https://lumingdong.cn/technology-evolution-trend-of-recommendation-system.html)



# 排序


# reference

https://tech.meituan.com/dl.html (深度学习在美团点评推荐平台排序中的运用)
https://tech.meituan.com/mt-recommend-practice.html (美团推荐算法实践)

https://tech.meituan.com/online-learning.html (Online Learning算法理论与实践)


https://zhuanlan.zhihu.com/p/25343518?utm_source=wechat_session&utm_medium=social （用深度学习（DNN）构建推荐系统 - Deep Neural Networks for YouTube Recommendations论文精读）


https://cloud.tencent.com/developer/article/1005416 (CTR 预估模型的进化之路)



https://www.cnblogs.com/mafeng/p/7912217.html (5类系统推荐算法,非常好使,非常全)


[推荐系统召回四模型之：全能的FM模型](https://zhuanlan.zhihu.com/p/58160982)

[推荐系统召回四模型之二：沉重的FFM模型](https://zhuanlan.zhihu.com/p/59528983)



https://juejin.im/post/5d75ad14f265da03c02c2827

http://www.broadview.com.cn/article/419572

https://zhuanlan.zhihu.com/p/93201318

https://mp.weixin.qq.com/s?__biz=MjM5ODYwMjI2MA==&mid=2649742233&idx=1&sn=f97cb1fcb3c4182168bdc5cef33bc9d3&chksm=bed348e289a4c1f40de9869c720cc9be2a027ca6919234fdfccec43d78c4b0e289296b630ca5&token=1629562894&lang=zh_CN&scene=21#wechat_redirect

https://toutiao.io/posts/uiy2ec/preview

https://www.jianshu.com/p/201b6029f247

https://www.ershicimi.com/p/a27ec89febf5c0ffd13b9fc7fb510439

https://ailab-aida.github.io/2019/11/19/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9/#more


