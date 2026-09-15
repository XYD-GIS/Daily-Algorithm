# GeoShapley

> [!IMPORTANT]
>
> This article introduces GeoShapley, a game theory approach to measuring spatial effects in machine learning models. GeoShapley extends the Nobel Prize–winning Shapley value framework in game theory by conceptualizing location as a player in a model prediction game, which enables the quantification of the importance of location and the synergies between location and other features in a model. GeoShapley is a model-agnostic approach and can be applied to statistical or black-box machine learning models in various structures. The interpretation of GeoShapley is directly linked with spatially varying coefficient models for explaining spatial effects and additive models for explaining non-spatial effects. Using simulated data, GeoShapley values are validated against known data-generating processes and are used for cross-comparison of seven statistical and machine learning models. An empirical example of house price modeling is used to illustrate GeoShapley’s utility and interpretation with real-world data. The method is available as an open source Python package named geoshapley.

# Shapley Basic

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

---

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

---

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

---

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

---

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

---

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

---

## 7. Shapley Value 与 Additive Model

Shapley 对单个 prediction 的解释：

$$
\hat y_i=\phi_0+\sum_{j=1}^{p}\phi_{ji}
$$

与 additive model（加性模型）有很强的联系。

例如线性回归：

$$
y=\beta_0+\beta_1X_1+\beta_2X_2
$$

也是：

$$
\text{Intercept}
+
X_1\text{ 的作用}
+
X_2\text{ 的作用}
$$

因此 SHAP 本质上提供了一种：

$$
\boxed{\text{Additive Explanation}}
$$

---

## 8. Figure 1：线性模型中的 Shapley Value

论文设定：

$$
y=3+2X_1+X_2
$$

且：

$$
X_1,X_2\sim U(0,4)
$$

因此：

$$
E(X_1)=E(X_2)=2
$$

模型的平均预测值为：

$$
\phi_0=E(y)=3+2\times2+2=9
$$

对于 $X_1$：

$$
\phi_1=2(X_1-2)
$$

对于 $X_2$：

$$
\phi_2=X_2-2
$$

因此：

- $\phi_1-X_1$ 关系的斜率为 2；
- $\phi_2-X_2$ 关系的斜率为 1；

恰好对应原线性模型中 $X_1$ 和 $X_2$ 的系数。

**核心理解：**

SHAP Value 表示某个特征相对于平均参考状态（baseline）对当前预测值产生的额外贡献，而不是该特征项本身的绝对值。

$$
\hat y_i=\phi_0+\sum_j\phi_{ji}
$$

在线性模型中可理解为：

$$
\boxed{
\phi_j=\beta_j(X_j-E[X_j])
}
$$

---

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

---

## 10. Monte Carlo Approximation（蒙特卡洛近似）

由于精确计算所有 feature combinations 成本很高，可以使用 Monte Carlo 方法近似 Shapley Value。

核心思想：

> 不计算所有 coalition，而是随机采样部分 coalition 或排列，利用这些样本估计平均边际贡献。

优点：

- 大幅降低计算量

缺点：

- 得到的是近似值
- 采样过少时估计可能不稳定

---

## 11. Kernel SHAP（核 SHAP）

Lundberg 和 Lee（2017）提出了 Kernel SHAP，它是一种 **model-agnostic（模型无关）** 的 Shapley Value 估计方法。

所谓模型无关：

> 不需要知道模型内部结构，只需要能够输入数据并得到 prediction。

因此原则上可以用于：

- Linear Regression
- SVM
- Neural Network
- Random Forest
- XGBoost 等

Kernel SHAP 的核心思想：

> 构造不同的 feature coalitions，观察每种 coalition 下模型 prediction 的变化，再通过 Shapley-weighted kernel 进行加权回归，从而估计各 feature 的 Shapley Value。

即：

$$
\text{feature combinations}
\rightarrow
\text{model predictions}
\rightarrow
\text{weighted regression}
\rightarrow
\text{Shapley Values}
$$

---

## 12. Kernel SHAP 的加权最小二乘

Kernel SHAP 可以转化为一个加权最小二乘回归问题：

$$
\phi_i=(Z^TWZ)^{-1}Z^TWv
$$

其中：

- $Z$：feature coalition 的二进制矩阵
  - 1：feature 存在
  - 0：feature 被 mask
  - 最后一列为 1，用于估计 intercept $\phi_0$
- $v$：每一种 coalition 下模型得到的 prediction
- $W$：Shapley kernel 权重矩阵
- $\phi_i$：第 $i$ 个样本对应的 Shapley Values

可以理解为：

$$
v
\approx
\phi_0+\phi_1z_1+\phi_2z_2+\cdots+\phi_pz_p
$$

即：

> 利用不同 feature combinations 下的模型预测结果，通过加权回归反推出各 feature 对 prediction 的贡献。

### Shapley Kernel 权重

对于 coalition $S$，其权重为：

$$
w_s=
\frac{p-1}
{\binom{p}{s}s(p-s)}
$$

其中：

- $p$：总 feature 数量
- $S$：某个 feature coalition
- $s=|S|$：coalition 中包含的 feature 数量
- $\binom{p}{s}$：从 $p$ 个 features 中选择 $s$ 个的组合数

核心思想：

> 较小和较大的 coalition 更容易分离出单个 feature 的贡献，因此获得更高权重。

---

## 13. Background Data（背景数据集）

Kernel SHAP 需要 background data set。

原因：

在 $Z$ 中：

- 1 表示 feature 存在
- 0 表示 feature 被 mask

但机器学习模型通常不能真正接受“feature 不存在”，因此被 mask 的 feature 必须用某个参考值进行替代。

Background Data 的作用：

> 用 background data 中的值替换被 mask 的 features，从而近似表示该 feature “不存在”的状态。

因此：

$$
\boxed{\text{mask} \neq \text{直接删除 feature}}
$$

而是：

$$
\boxed{\text{mask} = \text{用 background value 替代}}
$$

Background Data 可以选择：

- 全部数据
- 随机抽取的部分样本
- 代表性样本
- 单个 reference value
- mean / median
- k-means 得到的代表样本

Background Data 越大：

- Shapley Value 估计方差通常越小
- 计算成本越高

核心理解：

> Background Data 定义了 SHAP 解释中的“参考状态”。

---

## 14. Tree SHAP（树模型 SHAP）

Tree SHAP 是一种专门用于 **tree-based models（树模型）** 的 Shapley Value 估计方法。

相比 Kernel SHAP，Tree SHAP 的主要优势是：

> **计算效率更高。**

但它只能用于树模型，例如：

- Decision Tree
- Random Forest
- Gradient Boosting Tree
- XGBoost

因此：

$$
\boxed{\text{Kernel SHAP：模型通用，但较慢}}
$$

$$
\boxed{\text{Tree SHAP：树模型专用，但更快}}
$$

Tree SHAP 除了计算普通 Shapley Value 外，还可以计算 **Shapley Interaction Effect（Shapley 交互效应）**。

---

## 15. Shapley Interaction Effect（Shapley 交互效应）

Shapley Interaction Effect 用于衡量：

> 两个 features 同时作用时，超过二者各自独立作用之和的额外贡献。

$$
\Delta_{i,j}
=
f(S\cup\{i,j\})
-f(S\cup\{i\})
-f(S\cup\{j\})
+f(S)
$$

其中：

- $i,j$：需要研究 interaction 的两个 features
- $S$：不包含 $i,j$ 的某个 coalition
- $f(S\cup\{i,j\})$：$i,j$ 同时加入后的结果
- $f(S\cup\{i\})$：只加入 $i$
- $f(S\cup\{j\})$：只加入 $j$
- $f(S)$：两者都没有加入时的结果

因此：

$$
\boxed{
\text{Interaction}
=
\text{两者共同作用}
-
\text{各自独立作用}
}
$$

如果：

$$
\Delta_{i,j}=0
$$

说明两个 features 没有额外的 interaction。

如果：

$$
\Delta_{i,j}\neq0
$$

说明两个 features 同时存在时产生了额外作用。

完整的 Shapley Interaction Value 为：

$$
\phi(i,j)
=
\sum_{S\subseteq M\setminus\{i,j\}}
\frac{s!(p-s-2)!}{2(p-1)!}
\Delta_{i,j}
$$

核心理解：

> 对 $i$ 和 $j$ 在各种 coalitions 下产生的 interaction contribution 进行加权平均。

这为 GeoShapley 后面计算：

$$
GEO\times X_j
$$

提供了理论基础。

---

# GeoShapley Principles

## 16. GeoShapley：Location as a Joint Player

GeoShapley 的核心思想：

> 将所有 location features 视为一个 joint player，而不是分别解释。

例如：

$$
GEO=(X,Y)
$$

其中 $X,Y$ 为地理坐标。

如果 location features 有 $g$ 个，则：

$$
|GEO|=g
$$

例如：

$$
GEO=(u,v)
$$

则：

$$
g=2
$$

这样做的原因是：

> 单独某一个 coordinate 或 location embedding 往往没有完整的现实空间意义，而这些 location features 共同描述的才是“位置”。

### Equation (7)：Intrinsic Location Effect

GeoShapley 中位置本身的贡献为：

$$
\phi_{GEO}
=
\sum_{S\subseteq M\setminus GEO}
\frac{s!(p-s-g)!}{(p-g+1)!}
[f(S\cup GEO)-f(S)]
$$

其中：

- $GEO$：所有 location features 构成的 joint player
- $g$：GEO 中包含的 location feature 数量
- $S$：不包含 GEO 的某个 coalition
- $s=|S|$

核心部分：

$$
f(S\cup GEO)-f(S)
$$

表示：

> GEO 作为一个整体加入 coalition 后，对模型 prediction 产生的边际贡献。

因此：

$$
\boxed{\phi_{GEO}=\text{位置本身对 prediction 的贡献}}
$$

$\phi_{GEO}$ 可以理解为：

> **Intrinsic Location Effect（内在位置效应）**

其解释类似于：

- spatial fixed effect
- GWR 中的 local intercept

即：

> $\phi_{GEO}$ 表示位置本身对 prediction 的贡献，类似于 GWR 中的 local intercept；

而 feature 与 location 的交互项 $\phi(GEO,j)$，才对应某个 feature 的作用随空间位置变化。

---

## 17. Equation (8)：Nonspatial / Location-Invariant Feature Effect

对于普通非空间 feature $X_j$：

$$
\phi_j
=
\sum_{S\subseteq M\setminus\{j\}}
\frac{s!(p-s-g)!}{(p-g+1)!}
\left[
f(S\cup\{j\})-f(S)
\right]
$$

其中：

$$
f(S\cup\{j\})-f(S)
$$

仍然表示：

> feature $j$ 加入 coalition 后产生的 marginal contribution。

因此：

$$
\boxed{
\phi_j
=
feature\ j\text{ 的位置不变贡献}
}
$$

也就是：

> **location-invariant effect（位置不变效应）**

每一个非空间 feature 都有自己的 $\phi_j$。

例如模型中有：

$$
GDP,\ POP,\ VIIRS
$$

那么分别有：

$$
\phi_{GDP},\quad
\phi_{POP},\quad
\phi_{VIIRS}
$$

如果数据共有 $n$ 个样本，则：

$$
\phi_j=
[
\phi_{j1},
\phi_{j2},
\dots,
\phi_{jn}
]
$$

其中：

$$
\phi_{ji}
$$

表示：

> 第 $i$ 个样本中，feature $j$ 对 prediction 的贡献。

---

## 18. Equation (9)：Location–Feature Interaction

$$
\phi(GEO,j)
=
\sum_{S\subseteq M\setminus\{GEO,j\}}
\frac{s!(p-s-g-1)!}{(p-g+1)!}
\Delta_{GEO,j}
$$

其中：

$$
\Delta_{GEO,j}
=
f(S\cup\{GEO,j\})
-f(S\cup GEO)
-f(S\cup\{j\})
+f(S)
$$

参数含义：

- $GEO$：位置这个 joint player
- $j$：某一个普通非空间 feature
- $S$：除 $GEO$ 和 $j$ 之外，其他 features 组成的某个 coalition
- $s=|S|$：coalition $S$ 中 feature 的数量
- $\Delta_{GEO,j}$：在给定 coalition $S$ 下，GEO 与 feature $j$ 的 interaction contribution

注意：

$$
f(S\cup\{GEO,j\})
$$

不是“$j$ 在 GEO 里面”。

而是表示：

> 在 coalition $S$ 的基础上，同时加入 GEO 和 feature $j$。

$\Delta_{GEO,j}$ 表示：

> 在给定 coalition $S$ 下，GEO 和 feature $j$ 同时存在时产生的额外交互作用。

可以理解为：

$$
\boxed{
\text{GEO和}j\text{共同作用}
-
\text{GEO独立作用}
-
j\text{独立作用}
}
$$

而：

$$
\phi(GEO,j)
$$

表示：

> GEO 与 feature $j$ 在所有 coalitions 下 interaction contribution 的加权平均。

核心理解：

$$
\boxed{
\phi(GEO,j)
=
feature\ j\text{ 的作用因位置不同而产生的额外变化}
}
$$

---

## 19. GeoShapley 与 GWR 空间变系数的关系

$\phi(GEO,j)$ 与 **Spatially Varying Coefficient（空间变系数）** 直接相关。

例如 GWR：

$$
y_i
=
\beta_0(u_i,v_i)
+
\beta_j(u_i,v_i)X_{ji}
$$

其中：

$$
\beta_j(u,v)
$$

表示：

> feature $j$ 在不同空间位置具有不同的 local coefficient。

GeoShapley 中：

$$
\phi(GEO,j)
$$

表示：

> feature $j$ 的作用由于位置不同而产生的额外变化。

因此：

$$
\boxed{
\phi(GEO,j)
\leftrightarrow
feature\ j\text{ 的 spatially varying component}
}
$$

需要区分三个量：

### 1. $\phi_{GEO}$

表示：

> 位置本身对 prediction 的贡献。

类似于 GWR 中：

$$
\beta_0(u,v)
$$

即 local intercept。

### 2. $\phi_j$

表示：

> feature $j$ 本身的 location-invariant effect。

### 3. $\phi(GEO,j)$

表示：

> feature $j$ 的作用由于地理位置不同而发生的额外变化。

因此：

$$
\boxed{
\phi_j+\phi(GEO,j)
}
$$

表示：

> feature $j$ 在当前位置上的总贡献。

---

## 20. 从 GeoShapley Value 恢复 GWR-type Local Coefficient

论文给出：

$$
\boxed{
\beta_j(u,v)
=
\frac{
\phi_j+\phi(GEO,j)
}{
X_j-E[X_j]
}
}
$$

其中：

- $\beta_j(u,v)$：feature $j$ 在位置 $(u,v)$ 上的局部系数
- $\phi_j$：feature $j$ 的位置不变贡献
- $\phi(GEO,j)$：feature $j$ 与 location 的空间 interaction contribution
- $X_j$：当前样本中 feature $j$ 的实际值
- $E[X_j]$：feature $j$ 的期望值 / 参考平均水平
- $X_j-E[X_j]$：当前 feature value 相对于参考平均水平的偏离

其中：

$$
\phi_j+\phi(GEO,j)
$$

表示：

> feature $j$ 在当前位置上的总贡献。

整个公式可以理解为：

$$
\boxed{
\text{Local coefficient}
=
\frac{
\text{feature 在当前位置的总贡献}
}{
\text{feature 相对于参考值的偏离}
}
}
$$

前面的线性模型中：

$$
\phi_j
=
\beta_j(X_j-E[X_j])
$$

因此：

$$
\beta_j
=
\frac{\phi_j}{X_j-E[X_j]}
$$

GeoShapley 加入 spatial interaction 后：

$$
\text{当前位置总贡献}
=
\phi_j+\phi(GEO,j)
$$

所以：

$$
\boxed{
\beta_j(u,v)
=
\frac{
\phi_j+\phi(GEO,j)
}{
X_j-E[X_j]
}
}
$$

这使 GeoShapley 的解释结果可以转换成类似 GWR 的 spatially varying coefficient。

---

## 21. GeoShapley 如何基于 Kernel SHAP 求解

GeoShapley 并没有重新设计一套完全不同的求解算法。

它仍然扩展自 Kernel SHAP，并采用：

> **weighted least squares（加权最小二乘）**

进行估计。

即：

$$
Z,\ W,\ v
\rightarrow
\text{Weighted Least Squares}
\rightarrow
\text{GeoShapley Values}
$$

主要变化发生在：

$$
Z
$$

矩阵。

原因有两个：

1. location features 被合并成一个 joint player：$GEO$
2. 需要额外估计 $GEO\times X_j$ interaction

---

## 22. GeoShapley 中的 $Z$ Matrix

普通 Kernel SHAP 中：

$$
Z
$$

的大小为：

$$
2^p\times(p+1)
$$

GeoShapley 中修改为：

$$
\boxed{
2^{p-g+1}\times(2p-2)
}
$$

### 为什么行数减少？

原来有 $p$ 个 features。

其中：

$$
g
$$

个属于 location features。

GeoShapley 将这 $g$ 个 location features 合并成一个：

$$
GEO
$$

所以真正参与 coalition 的 players 数量变成：

$$
\boxed{
p-g+1
}
$$

因此 coalition 数量变成：

$$
\boxed{
2^{p-g+1}
}
$$

例如：

$$
p=4,\quad g=2
$$

原始 features：

$$
X_1,\ X_2,\ u,\ v
$$

普通 SHAP 有：

$$
2^4=16
$$

种 combinations。

GeoShapley 将：

$$
(u,v)\rightarrow GEO
$$

因此 players 变成：

$$
X_1,\ X_2,\ GEO
$$

所以：

$$
2^3=8
$$

种 combinations。

### 为什么列数增加？

因为 GeoShapley 除了估计：

$$
\phi_1,\phi_2,\phi_{GEO}
$$

还需要额外估计：

$$
\phi(GEO,1),\quad
\phi(GEO,2)
$$

也就是：

$$
GEO\times X_j
$$

的 interaction effect。

因此：

> 行数由于多个 location features 合并为 GEO 而减少；  
> 列数由于增加 interaction terms 而增加。

### Table 2 示例

假设：

$$
p=4,\quad g=2
$$

原始 features 为：

$$
X_1,\ X_2,\ u,\ v
$$

其中：

$$
GEO=(u,v)
$$

则 $Z$ 中包含：

- $X_1$
- $X_2$
- $X_1\times GEO$
- $X_2\times GEO$
- $GEO$
- Intercept

| Feature set   | $X_1$ | $X_2$ | $X_1\times GEO$ | $X_2\times GEO$ | $GEO$ | Intercept |
| ------------- | ----: | ----: | --------------: | --------------: | ----: | --------: |
| None          |     0 |     0 |               0 |               0 |     0 |         1 |
| $X_1$         |     1 |     0 |               0 |               0 |     0 |         1 |
| $X_2$         |     0 |     1 |               0 |               0 |     0 |         1 |
| $GEO$         |     0 |     0 |               0 |               0 |     1 |         1 |
| $X_1,GEO$     |     1 |     0 |               1 |               0 |     1 |         1 |
| $X_2,GEO$     |     0 |     1 |               0 |               1 |     1 |         1 |
| $X_1,X_2$     |     1 |     1 |               0 |               0 |     0 |         1 |
| $X_1,X_2,GEO$ |     1 |     1 |               1 |               1 |     1 |         1 |

需要注意：

> interaction 只有在两个对应 players 都存在时才为 1。

例如：

$$
X_1=1,\quad GEO=1
$$

才有：

$$
X_1\times GEO=1
$$

如果：

$$
GEO=0
$$

那么即使：

$$
X_1=1
$$

也有：

$$
X_1\times GEO=0
$$

---

## 23. Equation (10)：GeoShapley 的最终分解

GeoShapley 最终将 prediction 分解为四个部分：

$$
\boxed{
\hat y
=
\phi_0
+
\phi_{GEO}
+
\sum_j\phi_j
+
\sum_j\phi(GEO,j)
}
$$

### 1. $\phi_0$：Base Value

$$
\phi_0
$$

是 background data 上的平均 prediction。

可以理解为：

> 模型解释的基础预测水平。

它相当于：

$$
\boxed{\text{global intercept}}
$$

### 2. $\phi_{GEO}$：Intrinsic Location Effect

$$
\phi_{GEO}
$$

表示：

> 位置本身对 prediction 的贡献。

即：

$$
\boxed{\text{Location Effect}}
$$

其解释类似于 GWR 中的 local intercept。

### 3. $\phi_j$：Location-Invariant Feature Effect

对于每一个普通 feature：

$$
X_j
$$

都有：

$$
\phi_j
$$

表示：

> feature $j$ 本身、不随位置变化的贡献。

即：

$$
\boxed{\text{Feature Main Effect}}
$$

### 4. $\phi(GEO,j)$：Spatially Varying Interaction Effect

$$
\phi(GEO,j)
$$

表示：

> feature $j$ 的作用由于位置不同而产生的额外变化。

即：

$$
\boxed{
\text{Location}\times\text{Feature Interaction}
}
$$

因此 Equation (10) 可以理解成：

$$
\boxed{
Prediction
=
Base
+
Location
+
Feature
+
Location\times Feature
}
$$

### 如果不存在空间效应

如果模型中没有 spatial effects：

$$
\phi_{GEO}=0
$$

并且：

$$
\phi(GEO,j)=0
$$

Equation (10) 就退化成：

$$
\hat y
=
\phi_0+\sum_j\phi_j
$$

也就是普通 SHAP 的 additive decomposition。

---

## 24. GeoShapley Principles 核心总结

GeoShapley 最终把模型解释拆成三类真正需要关注的效应。

### 1. Intrinsic Location Effect

$$
\boxed{\phi_{GEO}}
$$

回答：

> **这个地方本身，对 prediction 有多大影响？**

### 2. Location-Invariant Feature Effect

$$
\boxed{\phi_j}
$$

回答：

> **feature $j$ 本身，对 prediction 有多大影响？**

### 3. Spatially Varying Interaction Effect

$$
\boxed{\phi(GEO,j)}
$$

回答：

> **feature $j$ 的作用，会因为位置不同而改变多少？**

因此：

$$
\boxed{
\phi_{GEO}
=
位置本身
}
$$

$$
\boxed{
\phi_j
=
特征本身
}
$$

$$
\boxed{
\phi(GEO,j)
=
位置对特征作用的调节
}
$$

最终：

$$
\boxed{
\hat y
=
\phi_0
+
\phi_{GEO}
+
\sum_j\phi_j
+
\sum_j\phi(GEO,j)
}
$$

---

## 25. 符号速查表

| 符号             | 含义                                                         |
| ---------------- | ------------------------------------------------------------ |
| $M$              | 所有 players / features 构成的集合                           |
| $S$              | 某一个 coalition                                             |
| $s$              | coalition $S$ 中 player / feature 的数量                     |
| $p$              | 原始 feature 总数                                            |
| $g$              | GEO 中 location features 的数量                              |
| $j$              | 某个普通非空间 feature                                       |
| $GEO$            | 所有 location features 构成的 joint player                   |
| $\phi_0$         | Base Value / Global Intercept                                |
| $\phi_j$         | feature $j$ 的 location-invariant contribution               |
| $\phi_{GEO}$     | intrinsic location effect                                    |
| $\Delta_{GEO,j}$ | 某一个 coalition 下 GEO 与 $j$ 的 interaction contribution   |
| $\phi(GEO,j)$    | GEO 与 feature $j$ 在所有 coalitions 下的加权平均 interaction |
| $Z$              | feature coalition 的 0/1 矩阵                                |
| $W$              | Shapley kernel 权重矩阵                                      |
| $v$              | 不同 coalition 下模型的 prediction                           |
| Background Data  | 定义被 mask feature 的参考状态                               |

---

## 26. 最终逻辑链

$$
\boxed{\text{Classic Shapley}}
$$

↓

计算单个 player 在所有 coalitions 下的平均 marginal contribution

↓

$$
\boxed{\text{SHAP}}
$$

↓

把 player 换成 feature，把 outcome 换成 model prediction

↓

$$
\hat y
=
\phi_0+\sum_j\phi_j
$$

↓

$$
\boxed{\text{Kernel SHAP}}
$$

↓

用 Shapley-weighted least squares 估计 $\phi$

↓

$$
\boxed{\text{GeoShapley}}
$$

↓

把 location features 合并成 joint player：

$$
GEO
$$

并加入：

$$
GEO\times X_j
$$

interaction

最终：

$$
\boxed{
\hat y
=
\phi_0
+
\phi_{GEO}
+
\sum_j\phi_j
+
\sum_j\phi(GEO,j)
}
$$
