[[TOC]]


# 推荐系统

## 推荐系统的目的

### 发掘长尾

理论基础是长尾理论，一小部分最热的资源得到大量的关注，而剩下的一大部分资源却无人问津，这不仅导致了资源的浪费，也让小众的用户无法找到自己感兴趣的内容

### 解决信息过载

提高信息利用率

### 提高点击率/转化率

### 个性化服务

## 推荐系统发展过程

- 嵌入(Embedding)和MLP范式: 原始特征嵌入到低维向量中，然后输入到MLP中以获得最终推荐结果。优势是可以连接不同的特征，缺点是忽略了用户行为的连续性。

- 使用Transformer模型捕获用户行为的序列信号

# ACM Recsys主要话题

## 推荐系统模型/效果评估

## 学术到工业的跨越问题

## 健壮性及推荐系统安全性


# 推荐系统应用场景

## 在线视频

## 电商网站

## 在线音乐

## 社交网络

# 推荐系统研究方向

## CTR预估

[华为若亚.2003.AutoFIS: Automatic Feature Interaction Selection in Factorization Models for Click-Through Rate Prediction](https://arxiv.org/abs/2003.11235)

介绍: 本文采用AutoML的搜索方法选择重要性高的二次特征交互项、去除干扰项，提升FM、DeepFM这类模型的准确率

[京东.2006.Category-Specific CNN for Visual-aware CTR Prediction at JD.com](https://arxiv.org/abs/2006.10337)

[Facebook.2207.Towards Automated Neural Interaction Discovering for Click-Through Rate Prediction](https://arxiv.org/abs/2007.06434)


### transformers在CTR中的应用

## TopN推荐

1. Dual Channel Hypergraph Collaborative Filtering 【百度】

笔记：https://blog.csdn.net/weixin_42052231/article/details/107710301
2. Probabilistic Metric Learning with Adaptive Margin for Top-K Recommendation 【华为诺亚】

3. Controllable Multi-Interest Framework for Recommendation 【阿里】

论文：https://arxiv.org/abs/2005.09347
4. Embedding-based Retrieval in Facebook Search 【Facebook】

论文：https://arxiv.org/abs/2006.11632
5. On Sampling Top-K Recommendation Evaluation



## 对话式推荐

1. Evaluating Conversational Recommender Systems via User Simulation

论文：https://arxiv.org/abs/2006.08732
2. Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion

论文：https://arxiv.org/abs/2007.04032
3. Interactive Path Reasoning on Graph for Conversational Recommendation

论文：https://arxiv.org/abs/2007.00194

## 序列推荐

1. Disentangled Self-Supervision in Sequential Recommenders 【阿里】

论文：http://pengcui.thumedialab.com/papers/DisentangledSequentialRecommendation.pdf
2. Handling Information Loss of Graph Neural Networks for Session-based Recommendation

3. Maximizing Cumulative User Engagement in Sequential Recommendation: An Online Optimization Perspective 【阿里】

论文：https://arxiv.org/pdf/2006.04520.pdf

## GNN


## 强化学习

1. Jointly Learning to Recommend and Advertise 【字节跳动】

论文：https://arxiv.org/abs/2003.00097
2. BLOB: A Probabilistic Model for Recommendation that Combines Organic and Bandit Signals 【Criteo】

3. Joint Policy-Value Learning for Recommendation 【Criteo】

论文：https://www.researchgate.net/publication/342437800_Joint_Policy-Value_Learning_for_Recommendation


## 迁移学习

1. Learning Transferrable Parameters for Long-tailed Sequential User Behavior Modeling 【Salesforce】

2. Semi-supervised Collaborative Filtering by Text-enhanced Domain Adaptation 【阿里】

论文：https://arxiv.org/abs/2007.07085

## 多任务学习

1. Privileged Features Distillation at Taobao Recommendations 【阿里】

论文：https://arxiv.org/abs/1907.05171



## AutoML

1. Neural Input Search for Large Scale Recommendation Models 【Google】

论文：https://arxiv.org/abs/1907.04471
2. Towards Automated Neural Interaction Discovering for Click-Through Rate Prediction 【Facebook】

论文：https://arxiv.org/abs/2007.06434

## 元学习

## Graph-based Recommendation

1. A Framework for Recommending Accurate and Diverse Items Using Bayesian Graph Convolutional Neural Networks 【华为诺亚】

2. An Efficient Neighborhood-based Interaction Model for Recommendation on Heterogeneous Graph 【Amazon】

论文：https://arxiv.org/abs/2007.00216
3. M2GRL: A Multi-task Multi-view Graph Representation Learning Framework for Web-scale Recommender Systems 【阿里】

简介：本文通过关联多个视角的图(item-item图、item-shop图、shop-shop图等)增强item表征，用于item召回。
论文：https://arxiv.org/abs/2005.10110
4. Handling Information Loss of Graph Neural Networks for Session-based Recommendation

5. Interactive Path Reasoning on Graph for Conversational Recommendation

论文：https://arxiv.org/abs/2007.00194
6. A Dual Heterogeneous Graph Attention Network to Improve Long-Tail Performance for Shop Search in E-Commerce 【阿里】

7. Gemini: A Novel and Universal Heterogeneous Graph Information Fusing Framework for Online Recommendations 【滴滴】


## Embedding and Representation

1. Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems 【Facebook】

论文：https://arxiv.org/abs/1909.02107
2. PinnerSage: Multi-Modal User Embedding Framework for Recommendations at Pinterest 【Pinterest】

论文：https://arxiv.org/abs/2007.03634
3. SimClusters: Community-Based Representations for Heterogeneous Recommendations at Twitter 【Twitter】

4. Time-Aware User Embeddings as a Service 【Yahoo】

论文：https://astro.temple.edu/~tuf28053/papers/pavlovskiKDD20.pdf


## Federated Learning

1. FedFast: Going Beyond Average for Faster Training of Federated Recommender Systems


## Evaluation

1. Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions 【Netflix, Spotify】

论文：https://arxiv.org/abs/2007.12986
2. Evaluating Conversational Recommender Systems via User Simulation

论文：https://arxiv.org/abs/2006.08732
3. On Sampled Metrics for Item Recommendation 【Google】

4. On Sampling Top-K Recommendation Evaluation



## Debiasing

1. Debiasing Grid-based Product Search in E-commerce 【Etsy】

论文：http://www.public.asu.edu/~rguo12/kdd20.pdf
2. Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions 【Netflix, Spotify】

论文：https://arxiv.org/abs/2007.12986
3. Attribute-based Propensity for Unbiased Learning in Recommender Systems: Algorithm and Case Studies 【Google】

论文：https://research.google/pubs/pub49273/
## POI Recommendation
1. Geography-Aware Sequential Location Recommendation 【Microsoft】

论文：http://staff.ustc.edu.cn/~liandefu/paper/locpred.pdf

Cold-Start Recommendation
1. MAMO: Memory-Augmented Meta-Optimization for Cold-start Recommendation

论文：https://arxiv.org/abs/2007.03183
2. Meta-learning on Heterogeneous Information Networks for Cold-start Recommendation

论文：https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=6158&context=sis_research

## Others
1. Improving Recommendation Quality in Google Drive 【Google】

论文：https://research.google/pubs/pub49272/
2. Temporal-Contextual Recommendation in Real-Time 【Amazon】

论文：https://assets.amazon.science/96/71/d1f25754497681133c7aa2b7eb05/temporal-contextual-recommendation-in-real-time.pdf


[推荐系统圈的高端聚会这十年在谈些什么？刑无刀](https://zhuanlan.zhihu.com/p/34083992)

TODO

[精选了 10 篇来自 SIGIR 2020、KDD 2020 等顶会的最新论文](https://www.jiqizhixin.com/articles/2020-06-22-6)

[AAAI2020推荐系统论文集锦](https://zhuanlan.zhihu.com/p/102949266)

[RSPapers](https://github.com/hongleizhang/RSPapers)


[前沿 | BERT在eBay推荐系统中的实践](https://mp.weixin.qq.com/s/tOVgW9cRsbbcRN5d6WPYPQ)

[Transformer 在推荐模型中的应用总结](https://www.infoq.cn/article/WMSD1O57YAeDxJC6kAk6?utm_source=related_read_bottom&utm_medium=article)

[阿里将 Transformer 用于淘宝电商推荐，效果优于 DIN 和谷歌 WDL](https://www.infoq.cn/article/OJvS7h8JXvD4XCW*qldw?utm_source=related_read_bottom&utm_medium=article)

[BERT 在 eBay 推荐系统中的实践(解决冷启动问题)](https://www.infoq.cn/article/dtexvmmvedzjyu7hgcxc)

[【推荐系统】近期必读的6篇顶会WWW2020【推荐系统】相关论文-Part3](https://posts.careerengine.us/p/5e95e0f679c80622c44d0292)

[KDD 2020 推荐系统论文一览](https://blog.csdn.net/abcdefg90876/article/details/108177820)

[I-2020新资源推荐算法一览、实践](https://blog.csdn.net/ngadminq/article/details/109234978)

[2020年推荐系统工程师炼丹手册](https://zhuanlan.zhihu.com/p/247941842)

[AAAI2020推荐系统论文集锦](https://zhuanlan.zhihu.com/p/102949266)

[近期有哪些值得读的推荐系统论文？来看看这份私人阅读清单](https://www.jiqizhixin.com/articles/2020-06-22-6)

[推荐算法最前沿 | KDD 2020推荐算法论文一览（内附下载链接）](https://cloud.tencent.com/developer/article/1748462)

[NLP预训练模型-Transformer：从原理到实战](https://blog.csdn.net/linxid/article/details/84321617)

[Transformer原理以及文本分类实战](https://blog.csdn.net/qq_36618444/article/details/106472126)

[图解Bert系列之Transformer实战](http://www.uml.org.cn/ai/2019101114.asp)

[Transformer论文翻译](https://blog.csdn.net/qq_29695701/article/details/88096455)

[如何快速跟进NLP领域最新技术？(文献阅读清单)](https://baijiahao.baidu.com/s?id=1634402434762249222&wfr=spider&for=pc)

[深度学习中的注意力模型（2017版）](https://zhuanlan.zhihu.com/p/37601161)

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


[TODO 深度学习推荐系统、CTR预估工业界实战论文整理分享](https://view.inews.qq.com/a/20201219A02ZJJ00)

[TODO 深度学习模型实战-深度学习模型在各大公司实际生产环境的应用讲解文章](https://github.com/DA-southampton/Tech_Aarticle)

[TODO 推荐！李宏毅《机器学习》国语课程(2021)上线！](https://zhuanlan.zhihu.com/p/353776276)

[TODO CTR预估模型发展历程(转)](https://zhuanlan.zhihu.com/p/104307718)

[TODO 目前工业界常用的推荐系统模型有哪些？](https://www.zhihu.com/question/314773668?sort=created)

[【CTR预估】互联网大厂CTR预估前沿进展 重要TODO](https://blog.csdn.net/fengdu78/article/details/112792061)

[谷歌、阿里、微软等 10 大深度学习 CTR 模型最全演化图谱【推荐、广告、搜索领域】重要TODO](https://www.6aiq.com/article/1557332028652)


