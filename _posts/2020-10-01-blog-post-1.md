---
title: '浅谈人工智能方法'
date: 2020-10-01
permalink: /posts/2020/10/blog-post-1/
tags:
  - cool posts
  - category1
  - category2
---
# 浅谈人工智能方法

[toc]

> 以下内容是我在申请博士期间所做的整理与思考 2020.10

## 卷积神经网络

计算机视觉（Computer Vision）是人工智能的重要组成部分，基于卷积神经网络（Convolutional Neural Network）被证明是最有效的方法。我认为其发展主要经历了以下几个阶段：

- 1959年Huble 和 Wiesel对于猫脑视觉中枢的生物学实验证明视觉是基于简单结构的
- 1970年David Marr所著《vision》书揭示了视觉是层次的（hierarchical）
- Kunihiko Fukushima的论文是的应用神经网络研究视觉。 
- 上世纪九十年代纽约大学的Yann Lecun应用反向传播学习（Backpropagation）和卷积神经网络（Convolutional  Neural Network）在Minist数据集上的数字分类成功标志着机器视觉领域逐渐走向成熟。
- 在新世纪人工智能爆发的元年是2012，当Alex Krizhevsky用于20年前Yann Lecun相近的模型结构AlexNet，依托于强大的计算能力，基于CNN和backpropagation的AlexNet给ImageNet图像分类问题的准确率带来了质的提升。

从2012年8层网络的AlexNet，到2014年大约20层网络的的GoogleNet 和VGG再到2015年152层的ResNet。 随着神经网络模型越来越“深”，模型的表达能力与准确性与越来越高，深度学习（Deep Learning）一词应运而生。

 

实践证明神经网络模型的表达能力是十分强大的，其在机器视觉领域的成功足以说明。反向传播学习使得人们拥有end to end训练模型的方法。区别于以往精心挑选的特征，神经网络模型+反向传播学习给予我们end to end的表达能力。换句话说，以往模型的好坏很大程度上取决于特征选取的好坏（例如 Histogram of Gradient等）， 如今深度学习模型可以通过对海量数据的学习与训练，自己找出特征。这种强大的模型表达能力为人工智能与机器学习领域注入了强大的动力。DeepMind的基于Q-Learning和Deep Learning的DQN模型和Alpha Go就是典型的例子。

 

## 生成对抗网络

生成对抗网络（Generative Adversarial Network）自从Ian Goodfellow于2014年提出以来， 其强大的对数据分布的学习与采样能力、以及良好的生成结果使得一经出现就收到广泛的关注。其核心思想是通过优化对抗损失函数同时训练生成器（Generator）和判别器（Discriminatory），使得生成器可以高效的学习数据的分布并可以进行采样。其理想情况是判别器无法判别图像是真是图像还是生成图像。

 

随着越来越多学者展开研究，GAN已经得到了长足的发展，无论是理论的还是模型的还是技巧的。

- 按照生成结果的角度，从最初的28*28的分辨率到后来DCGAN 64*64分辨率再到2017年PGGAN 1024*1024的超高十分真实的分辨率，GAN生成图片的质量与大小越来越高。

- 按照网络结构设计的角度，由最初的multilayer perceptron 到引入卷积神经网络再到引入Self-attention、coarse to fine等结构，GAN的模型也变得越来越复杂可以解决越来越多问题 。

- 按照优化损失函数的角度，Wasserstein Distance等度量的引入使得GAN的训练更加稳定。

- 按照研究方向，利用GAN做数据增强来做半监督学习问题、基于conditional GAN的风格转换问题、图像生成、超分辨率等等。

同时GAN的训练不稳定性、mode collapse问题、“黑箱子”问题等一系列问题仍然等待着我们去不断完善与解决。

 

有意思的是，这种基于对抗的学习方法在很多领域都有不同的体现。比如在强化学习中actor critic就有些对抗的思想。有的学者基于GAN的思想，用强化学习中的reward而不是loss function来训练网络。同时也有的学者让两个神经网络进行“交流”从而研究语言的产生。我有一个不成熟但大胆的假设，这种基于互相交流（对抗的、或者合作的）、多个的神经网络表示的智能体（agent）的研究也许是未来人工智能发展的一个方向，同时可以让我们更好的研究群体智能问题。



##  强化学习

强化学习（Reinforcement Learning）是机器学习的一大类问题，其主要区别于监督学习和无监督学习的地方是“监督”的滞后性。通过智能体（agent）在环境（environment）中不断试错与学习 （trial and error）的方式学习价值函数（value function），最终学习最有策略。其底层的数学模型为动态控制理论（Dynamic Control）中的马尔科夫决策过程（Markov Decision Process）。其理论在过去几十年里得到了长足的发展。比如基于模型（model based）的Dynamic Programming、Policy iteration和Value Iteration。 无模型预测（model-free prediction）中的Monte-Carlo Learning, Temporal Difference Learning等。无模型控制（model-free control）领域中的Q-learning，SARSA等。直接用函数去近似价值函数（value function）和策略（policy）对于模型的扩展性和泛化拥有很大的价值。此类方法被称为Value Function Approximation 和Policy Gradient。 

 

深度强化模型（Deep Reinforcement Learning）就是基于深度学习与强化学习发展出来的方法。其核心思想是用神经网络模型对价值函数和策略进行建模，通过随机梯度下降的方法进行学习并寻找最优策略。当前，无论从机器人控制到游戏AI再到自动驾驶，深度强化学习都有长足的发展。比如上文中所述的用DQN模型训练玩Atari游戏和Alpha Go学习下围棋。

 

如果我们将以上的方法视为一个智能体agent去学习环境从而得到最佳策略的话，一个自然的拓展就是：可不可同时让多个智能体去学习？比如在星际争霸（StarCraft）游戏中，除了让一个智能体去控制所有的士兵，能不能让多个智能体去控制他们，从而学习到更加好的配合策略？这属于多智能体强化学习（Multi-agent Reinforcement Learning）的问题了。更进一步的，有没有一种基于群体智能（Swarm Intelligence）的方法，通个每个微观上独立的、简单的智能体的学习，从宏观上涌现（Emergent）出智能的方法？

 

多智能体强化学习是强化学习的子问题。我个人粗略的把目前多智能体强化学习分为以下三类：集中的（centralized）、分散（decentralized）的的和分层的（hierarchical）。所谓集中是指每个智能体都共享相同的参数；分层是指有一个上级控制每个下级智能体，下级智能体可以有不同的参数，但需要将学习的数据反馈给上级；分散是指每一个智能体之间独立学习没有上级控制。其中集中式的方法是主流，所谓集中是指每个智能体都共享相同的参数。分层式的方法逐渐开始被重视，分散式的研究还处于初步阶段。

