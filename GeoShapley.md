# GeoShapley

> [!IMPORTANT]
>
> This article introduces GeoShapley, a game theory approach to measuring spatial effects in machine learning models. GeoShapley extends the Nobel Prize–winning Shapley value framework in game theory by conceptualizing location as a player in a model prediction game, which enables the quantification of the importance of location and the synergies between location and other features in a model. GeoShapley is a model-agnostic approach and can be applied to statistical or black-box machine learning models in various structures. The interpretation of GeoShapley is directly linked with spatially varying coefficient models for explaining spatial effects and additive models for explaining non-spatial effects. Using simulated data, GeoShapley values are validated against known data-generating processes and are used for cross-comparison of seven statistical and machine learning models. An empirical example of house price modeling is used to illustrate GeoShapley’s utility and interpretation with real-world data. The method is available as an open source Python package named geoshapley.

# Shaplely Basic

## 1. Shapley Value 是什么

==相对于 baseline，它多贡献了多少==

Shapley Value起源于合作博弈论（coalitional game theory），用于公平地分配不同参与者（players）对最终结果的贡献。

核心思想：

> 一个参与者的 Shapley Value，是其在所有可能联盟（coalitions）中的边际贡献（marginal contribution）的加权平均。

在机器学习中可以对应为：

- Player → Feature
- Game → Prediction model
- Outcome → Model prediction
- Contribution → Feature contribution to prediction

## 2. Shapley Value 公式

$$
\phi_j =
\sum_{S\subseteq M\setminus\{j\}}
\frac{s!(p-s-1)!}{p!}
\left[f(S\cup\{j\})-f(S)\right]
$$

其中：

- $p$：全部 players 的数量
- $j$：当前要计算贡献的 player
- $S$：不包含 $j$ 的某个 player coalition
- $s=|S|$：coalition $S$ 中 player 的数量
- $f(S)$：coalition $S$ 的结果
- $f(S\cup\{j\})$：在 $S$ 中加入 $j$ 后的结果

其中最重要的是：

$$
f(S\cup\{j\})-f(S)
$$

表示 player $j$ 在 coalition $S$ 下的 **边际贡献（marginal contribution）**。

## 3. 三个 Player 的例子

论文设置三个 players：

| Player set $S$ | Outcome $f(S)$ |
| -------------- | -------------: |
| None           |              0 |
| A              |              5 |
| B              |             10 |
| C              |            100 |
| A, B           |              5 |
| A, C           |            120 |
| B, C           |            140 |
| A, B, C        |            150 |

以 A 为例：

$$
\phi_A
=
\frac{f(A)-f(None)}{3}
+
\frac{f(A,B)-f(B)}{6}
+
\frac{f(A,C)-f(C)}{6}
+
\frac{f(A,B,C)-f(B,C)}{3}
$$

代入：

$$
\phi_A
=
\frac{5}{3}
+
\frac{-5}{6}
+
\frac{20}{6}
+
\frac{10}{3}
=
7.5
$$

同理：

$$
\phi_B=20
$$

$$
\phi_C=122.5
$$

因此 C 对最终结果的贡献最大。

**理解：**

==Shapley Value 不是只看“删除某个变量后结果下降多少”，而是考虑这个变量加入所有可能特征组合时所产生的边际贡献。==

## 4. Shapley Value 的四个性质

### 1. Efficiency（效率性）

所有 players 的 Shapley Value 之和等于最终总结果。

例如：

$$
7.5+20+122.5=150
$$

### 2. Symmetry（对称性）

如果两个 players 在所有 coalition 中的边际贡献都相同，则二者的 Shapley Value 相同。

### 3. Null Player（零玩家性）

如果某个 player 在所有 coalition 中的边际贡献始终为 0，则：

$$
\phi_j=0
$$

### 4. Additivity（可加性）

如果两个 game 相加，则 player 在组合 game 中的 Shapley Value 等于它在两个 game 中 Shapley Value 的和。

## 5. 从 Shapley Value 到机器学习解释

Shapley Value 可以用于解释单个模型预测。

对于第 $i$ 个样本：

$$
\hat y_i
=
\phi_0
+
\sum_{j=1}^{p}\phi_{ji}
$$

其中：

- $\hat y_i$：模型对第 $i$ 个样本的预测值
- $\phi_0$：base value
- $\phi_{ji}$：feature $j$ 对第 $i$ 个样本预测的 Shapley Value

因此：

> 每一个预测值都可以拆解成“基准值 + 每个特征的贡献”。

例如：

$$
\phi_0=80
$$

$$
\phi_1=+20,\quad
\phi_2=+10,\quad
\phi_3=-5
$$

则：

$$
\hat y=80+20+10-5=105
$$

正 SHAP Value 表示该特征将预测值相对于 baseline 向上推动；
负 SHAP Value 表示向下推动。

## 6. Shapley Value 的计算问题与 SHAP

经典 Shapley Value 需要考虑所有 feature combinations。

如果有 $p$ 个 features：

$$
2^p
$$

种 combinations。

例如：

$$
2^{15}=32768
$$

因此特征较多时计算量会迅速增加。

常见近似或高效算法：

- Monte Carlo approximation
- Kernel SHAP
- Tree SHAP

### Kernel SHAP

Kernel SHAP 是一种 model-agnostic 方法。

核心思想：

通过一个 Shapley-weighted 的加权最小二乘回归来估计 Shapley Value：

$$
\phi_i
=
(Z^TWZ)^{-1}Z^TWv
$$

其中：

- $Z$：特征组合矩阵，1 表示 feature 存在，0 表示 feature 缺失
- $v$：不同 feature combinations 下模型的预测结果
- $W$：Shapley kernel 权重
- $\phi_i$：样本 $i$ 的 Shapley Values

### Background Data

Kernel SHAP 需要 background dataset。

原因：

模型中的 feature 被 mask 后，需要使用 background data 中的值来代表该 feature 的“缺失状态”。

Background data 可以使用：

- 全部样本
- 随机样本
- 代表性样本
- mean / median
- k-means 聚类得到的代表样本

Background data 越大：
- Shapley Value 估计方差通常越小
- 计算成本越高

### Shapley Interaction

Shapley interaction 用来衡量两个 features 联合作用产生的额外贡献。

$$
\Delta_{i,j}
=
f(S\cup\{i,j\})
-f(S\cup\{i\})
-f(S\cup\{j\})
+f(S)
$$

理解：

> 将 feature $i$ 和 feature $j$ 各自独立产生的贡献扣除后，剩下的就是两者的 interaction effect。

## 9. Shapley Value 的计算复杂度

经典 Shapley Value 需要考虑大量 feature combinations。

对于 $p$ 个 features：

$$
2^p
$$

种 combinations。

例如：

$$
2^{15}=32768
$$

因此，随着 feature 数量增加，计算复杂度呈指数增长。

核心问题：

> Shapley 的计算瓶颈来自组合爆炸（combinatorial explosion）。

## 10. Monte Carlo Approximation（蒙特卡洛近似）

由于精确计算所有 feature combinations 成本很高，可以使用 Monte Carlo 方法近似 Shapley Value。

核心思想：

> 不计算所有 coalition，而是随机采样部分 coalition 或排列，利用这些样本估计平均边际贡献。

优点：

- 大幅降低计算量

缺点：

- 得到的是近似值
- 采样过少时估计可能不稳定
