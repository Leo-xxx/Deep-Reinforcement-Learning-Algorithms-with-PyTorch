## 17 种深度强化学习算法的 Pytorch 实现

[AI有道](javascript:void(0);) *前天*

点击上方“**AI有道**”，选择“星标”公众号

重磅干货，第一时间送达![img](https://mmbiz.qpic.cn/mmbiz_jpg/ow6przZuPIENb0m5iawutIf90N2Ub3dcPuP2KXHJvaR1Fv2FnicTuOy3KcHuIEJbd9lUyOibeXqW8tEhoJGL98qOw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



![img](https://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb3VZMubzVSKy1P46K3KWb56DUgh8d6sAjh52icetQTCwSjgafgnQDnbQo7ibNrqfRIWPdFiaoHR1hr6w/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)













来源：github

转自：新智元

编辑：肖琴



##### 深度强化学习已经在许多领域取得了瞩目的成就，并且仍是各大领域受热捧的方向之一。本文推荐一个用PyTorch实现了17种深度强化学习算法的教程和代码库，帮助大家在实践中理解深度RL算法。



深度强化学习已经在许多领域取得了瞩目的成就，并且仍是各大领域受热捧的方向之一。本文推荐一个包含了 17 种深度强化学习算法实现的 PyTorch 代码库。



![img](https://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb3VZMubzVSKy1P46K3KWb56lx3mzcCl6iamdfBOvicbIW2Gvg9zAzeow11s1AE18LCzaHoOeGO2nUSg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)









已实现的算法包括：





1. Deep Q Learning (**DQN**) (Mnih et al. 2013)
2. **DQN with Fixed Q Targets** (Mnih et al. 2013)
3. Double DQN (**DDQN**) (Hado van Hasselt et al. 2015)
4. **DDQN with Prioritised Experience Replay** (Schaul et al. 2016)
5. **Dueling DDQN** (Wang et al. 2016)
6. **REINFORCE** (Williams et al. 1992)
7. Deep Deterministic Policy Gradients (**DDPG**) (Lillicrap et al. 2016 )
8. Twin Delayed Deep Deterministic Policy Gradients (**TD3**) (Fujimoto et al. 2018)
9. Soft Actor-Critic (**SAC & SAC-Discrete**) (Haarnoja et al. 2018)
10. Asynchronous Advantage Actor Critic (**A3C**) (Mnih et al. 2016)
11. Syncrhonous Advantage Actor Critic (**A2C**)
12. Proximal Policy Optimisation (**PPO**) (Schulman et al. 2017)
13. DQN with Hindsight Experience Replay (**DQN-HER**) (Andrychowicz et al. 2018)
14. DDPG with Hindsight Experience Replay (**DDPG-HER**) (Andrychowicz et al. 2018 )
15. Hierarchical-DQN (**h-DQN**) (Kulkarni et al. 2016)
16. Stochastic NNs for Hierarchical Reinforcement Learning (**SNN-HRL**) (Florensa et al. 2017)
17. Diversity Is All You Need (**DIAYN**) (Eyensbach et al. 2018)





所有的实现都能够快速解决 **Cart Pole** (离散动作)、 **Mountain Car** (连续动作)、 **Bit Flipping** (动态目标的离散动作) 或 **Fetch Reach** (动态目标的连续动作) 等任务。本 repo 还会添加更多的分层 RL 算法。







已实现的环境：



1. Bit Flipping 游戏 (Andrychowicz et al. 2018)
2. Four Rooms 游戏 (Sutton et al. 1998)
3. Long Corridor 游戏 (Kulkarni et al. 2016)
4. Ant-{Maze, Push, Fall} (Nachum et al. 2018)

















**结果**







**1. Cart Pole 和 Mountain Car**







下面展示了各种 RL 算法成功学习离散动作游戏 Cart Pole 或连续动作游戏 Mountain Car 的结果。使用 3 个随机种子运行算法的平均结果如下图所示，阴影区域表示正负 1 标准差。使用的超参数可以在 results/cart_pol .py 和 results/Mountain_Car.py 文件中找到。







![img](https://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb3VZMubzVSKy1P46K3KWb56Uc70nejibPH1QlwHNkWGPPKt1Mgr8WbTXPGR1txfNspLGTYJMCtTQjA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)







**2. 事后经验重演 (HER) 实验**







下面展示了 DQN 和 DDPG 在 Bit Flipping (14 bits) 和 Fetch Reach 环境中的表现，这些环境在论文 Hindsight Experience Replay 和 Multi-Goal Reinforcement Learning 中有详细描述。这些结果复现了论文中发现的结果，并展示了添加 HER 可以如何让一个 agent 解决它原本无法解决的问题。请注意，在每对 agents 中都使用了相同的超参数，因此它们之间的唯一区别是是否使用了 hindsight。







![img](https://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb3VZMubzVSKy1P46K3KWb56Xo46YibibFs1YUgbV7Tlm6vD5dpAJV5N3YcIhlyYgicBgUq8S2cKgkDhA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)







**3. 分层强化学习实验**







下图左边的结果显示了在 Long Corridor 环境中 DQN 和 Kulkarni 等人在 2016 年提出的 hierarchy -DQN 算法的性能。该环境要求 agent 在返回之前走到走廊的尽头，以便获得更大的奖励。这种延迟满足和状态的混叠使得它在某种程度上是 DQN 不可能学习的游戏，但是如果我们引入一个元控制器 (如 h-DQN) 来指导低层控制器如何行动，就能够取得更大的进展。这与论文中发现的结果一致。







下图右边的结果显示了 Florensa 等人 2017 年提出的 DDQN 算法和用于分层强化学习的随机神经网络 (SNN-HRL) 的性能。使用 DDQN 作为比较，因为 SSN-HRL 的实现使用了其中的 2 种 DDQN 算法。







![img](https://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb3VZMubzVSKy1P46K3KWb56DrAZ67ia5194yIePymC6Fwiatu2CBqs4yM4kG0VOzDmfIsWVfr0bsstg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)









**用法**







存储库的高级结构是：

- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 

```
├── agents                        ├── actor_critic_agents       ├── DQN_agents             ├── policy_gradient_agents    └── stochastic_policy_search_agents ├── environments   ├── results                 └── data_and_graphs        ├── tests├── utilities                 └── data structures
```









**i) 观看智能体学习上述游戏**







观看所有不同的智能体学习 Cart Pole，请遵循以下步骤：







- 
- 
- 
- 
- 
- 
- 
- 
- 
- 

```
git clone https://github.com/p-christ/Deep_RL_Implementations.gitcd Deep_RL_Implementationsconda create --name myenvnameyconda activate myenvnamepip3 install -r requirements.txtpython Results/Cart_Pole.py
```

对于其他游戏，将最后一行更改为结果文件夹中的其他文件就行。







**ii) 训练智能体实现另一种游戏**







Open AI gym 上的环境都是有效的，你所需要做的就是更改 config.environment 字段。







如果你创建了一个继承自 gym.Env 的单独类，那么还可以使用自己的自定义游戏。请参阅 Environments/Four_Rooms_Environment.py 自定义环境的示例，然后查看脚本 Results/Four_Rooms.py 了解如何让 agents 运行环境。



GitHub地址：

https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch