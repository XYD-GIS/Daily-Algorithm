# Spatial Econometric Model & Diagnostic Tests

# SDEM

## 1. 从 OLS 开始理解 SDEM

普通 OLS：

$$
Y=X\beta+\varepsilon
$$

其中：

- $Y$：因变量
- $X$：解释变量
- $\beta$：回归系数
- $\varepsilon$：OLS 没有解释掉的部分，即误差项

OLS 的基本思想是：

$$
\boxed{
本地区的X
\rightarrow
本地区的Y
}
$$

例如：

$$
Heritage_i
=
\beta_0
+
\beta_1POP_i
+
\beta_2GDP_i
+
\varepsilon_i
$$

表示：

> 地区 $i$ 的人口、GDP 等因素影响地区 $i$ 的不可移动文物数量。

但空间数据存在一个重要特点：

> **一个地区通常不是孤立的，周围地区也可能对它产生影响。**

因此需要进一步考虑 spatial dependence（空间依赖）。

---

## 2. SDEM 是什么

SDEM 全称：

> **Spatial Durbin Error Model**
>
> 空间杜宾误差模型

标准形式可以写为：

$$
\boxed{
Y
=
X\beta
+
WX\theta
+
\varepsilon
}
$$

同时：

$$
\boxed{
\varepsilon
=
\lambda W\varepsilon
+
\mu
}
$$

因此 SDEM 可以理解为：

$$
\boxed{
SDEM
=
OLS
+
邻近地区X的影响
+
空间相关误差
}
$$

也可以写成：

$$
\boxed{
Y
=
本地因素
+
邻近地区因素
+
空间相关的遗漏因素
+
随机误差
}
$$

---

## 3. SDEM 中的 $W$ 是什么

$W$ 表示：

> **Spatial Weight Matrix（空间权重矩阵）**

它用于描述不同地区之间的空间邻接关系。

例如：

$$
A-B-C
$$

如果：

- A 和 B 相邻
- B 和 C 相邻
- A 和 C 不相邻

则空间权重矩阵可以表示这种邻接关系。

常见空间权重包括：

- Queen Contiguity
- Rook Contiguity
- Distance-based Weight
- K-nearest Neighbors

其中 Queen Contiguity 表示：

> 两个空间单元只要共享边或顶点，就认为二者相邻。

---

# $WX$：邻近地区解释变量的作用

## 4. 什么是 $WX$

SDEM 中：

$$
WX
$$

表示：

> **邻近地区解释变量的空间加权值。**

例如：

$$
WPOP_i
$$

表示：

> 地区 $i$ 周围邻近地区人口的加权水平。

因此：

$$
Y
=
X\beta
+
WX\theta
+
\varepsilon
$$

可以拆成：

$$
X\beta
$$

表示：

> 本地区自身因素的作用。

而：

$$
WX\theta
$$

表示：

> 邻近地区因素对本地区产生的影响。

---

## 5. $X$ 和 $WX$ 的区别

例如研究：

$$
Y=\text{不可移动文物数量}
$$

解释变量：

$$
X=POP
$$

那么：

$$
POP_i
$$

表示：

> 地区 $i$ 自己的人口。

而：

$$
WPOP_i
$$

表示：

> 地区 $i$ 周围地区的人口水平。

因此：

$$
Heritage_i
=
\beta POP_i
+
\theta WPOP_i
+
\varepsilon_i
$$

可以理解为：

> 一个地区的不可移动文物分布不仅可能与本地区人口有关，也可能与周围地区的人口条件有关。

其中：

$$
\beta
$$

反映本地效应；

$$
\theta
$$

反映邻近地区变量产生的空间作用。

---

# Spatial Error

## 6. 什么是误差项 $\varepsilon$

OLS 中：

$$
Y=X\beta+\varepsilon
$$

其中：

$$
\varepsilon
$$

不是“数据出错”。

而是：

> **模型中的解释变量解释完以后，仍然没有被解释的部分。**

即：

$$
\boxed{
Residual
=
Observed
-
Predicted
}
$$

例如：

某地区实际有：

$$
100
$$

处文物。

模型预测：

$$
80
$$

处。

则残差为：

$$
\varepsilon=100-80=20
$$

也就是说：

> 模型低估了 20。

---

## 7. 什么是空间误差

普通 OLS 通常希望：

> 不同地区的误差彼此独立。

例如：

$$
\varepsilon_A
$$

和：

$$
\varepsilon_B
$$

之间不存在系统性关系。

但空间数据中可能出现：

| 地区 | 实际值 | 预测值 | Residual |
| ---- | -----: | -----: | -------: |
| A    |    100 |     80 |      +20 |
| B    |    120 |     95 |      +25 |
| C    |     90 |     70 |      +20 |

如果 A、B、C 又恰好是相邻地区：

$$
A-B-C
$$

那么会发现：

> 相邻地区都被模型系统性低估。

即：

$$
\varepsilon_A>0
$$

$$
\varepsilon_B>0
$$

$$
\varepsilon_C>0
$$

并且这些正残差在空间上聚集。

这说明：

$$
\boxed{
Residuals\ are\ spatially\ correlated
}
$$

即：

> **残差存在空间自相关。**

---

## 8. 为什么会产生空间误差

假设模型只有：

$$
Heritage
=
POP
+
GDP
+
Road
+
\varepsilon
$$

但真实世界中还有一些没有进入模型的因素，例如：

- 历史文化核心区
- 古代都城体系
- 历史交通通道
- 地域文化圈
- 地方保护政策
- 未观测的自然环境条件

这些因素可能具有明显的空间集聚特征。

例如：

> 西安、咸阳、渭南可能同时受到某种历史区域因素影响。

如果这个因素没有进入模型，就会进入：

$$
\varepsilon
$$

于是相邻地区可能同时出现较大的正残差或负残差。

因此：

$$
\boxed{
空间误差
\neq
测量错误
}
$$

而更应该理解为：

$$
\boxed{
模型遗漏掉的空间性影响
}
$$

---

## 9. SDEM 如何处理空间误差

SDEM 中：

$$
\boxed{
\varepsilon
=
\lambda W\varepsilon
+
\mu
}
$$

其中：

- $\varepsilon$：包含空间结构的误差项
- $W\varepsilon$：邻近地区误差的空间加权值
- $\lambda$：Spatial Error Coefficient
- $\mu$：真正随机的误差

因此原来的误差：

$$
\varepsilon
$$

被进一步拆成：

$$
\boxed{
\varepsilon
=
空间相关部分
+
随机部分
}
$$

即：

$$
\boxed{
\varepsilon
=
\lambda W\varepsilon
+
\mu
}
$$

---

## 10. $\lambda$ 的含义

$$
\lambda
$$

表示：

> 误差项之间空间相关的强度。

如果：

$$
\lambda>0
$$

说明：

> 相邻地区的残差倾向于具有相同方向。

例如：

- 正残差附近仍然多为正残差
- 负残差附近仍然多为负残差

如果：

$$
\lambda=0
$$

则：

$$
\varepsilon=\mu
$$

表示：

> 误差中不存在需要考虑的空间相关结构。

---

## 11. $WX$ 与 $W\varepsilon$ 一定要区分

### $WX$

表示：

$$
\boxed{
看得见的邻居影响
}
$$

例如：

$$
WPOP
$$

就是邻近地区的人口。

它回答：

> **周围地区的解释变量，会不会影响本地区？**

---

### $W\varepsilon$

表示：

$$
\boxed{
没有被模型显式观测到的空间影响
}
$$

它回答：

> **模型没有解释掉的部分，是否在空间上仍然具有规律？**

因此：

$$
\boxed{
WX
=
Observed\ Spatial\ Spillover
}
$$

而：

$$
\boxed{
W\varepsilon
=
Unobserved\ Spatial\ Dependence
}
$$

---

## 12. SDEM 的核心理解

普通 OLS：

$$
\boxed{
Y
=
本地区X
+
未解释部分
}
$$

SDEM：

$$
\boxed{
Y
=
本地区X
+
周围地区X
+
空间相关遗漏因素
+
随机误差
}
$$

最重要的一句话：

> **SDEM 不是在 OLS 中简单增加一个误差，而是将原来 OLS 中的误差进一步分解，并同时考虑邻近地区解释变量产生的空间作用。**

---

# Direct / Indirect Effects

## 13. Direct Effect

Direct Effect 表示：

> 本地区的解释变量变化对本地区因变量产生的影响。

例如：

$$
POP_i
\rightarrow
Heritage_i
$$

即：

> 本地区人口变化与本地区不可移动文物分布之间的关系。

---

## 14. Indirect Effect

Indirect Effect 又称：

> **Spatial Spillover Effect（空间溢出效应）**

表示：

> 周围地区变量的变化对目标地区产生的影响。

例如：

$$
WPOP_i
\rightarrow
Heritage_i
$$

即：

> 周围地区的人口条件对地区 $i$ 的文物分布产生的作用。

---

## 15. Total Effect

总效应：

$$
\boxed{
Total\ Effect
=
Direct\ Effect
+
Indirect\ Effect
}
$$

即：

$$
\boxed{
总效应
=
本地作用
+
空间溢出作用
}
$$

---

# OLS / SEM / SLX / SDM / SDEM

## 16. 不同空间模型的区别

| Model | Formula                                                      | 核心思想                 |
| ----- | ------------------------------------------------------------ | ------------------------ |
| OLS   | $Y=X\beta+\varepsilon$                                       | 只考虑本地 $X$           |
| SEM   | $Y=X\beta+\varepsilon$；$\varepsilon=\lambda W\varepsilon+\mu$ | 处理空间相关误差         |
| SLX   | $Y=X\beta+WX\theta+\varepsilon$                              | 考虑邻近地区 $X$         |
| SDM   | $Y=\rho WY+X\beta+WX\theta+\varepsilon$                      | 同时考虑 $WY$ 和 $WX$    |
| SDEM  | $Y=X\beta+WX\theta+\varepsilon$；$\varepsilon=\lambda W\varepsilon+\mu$ | 同时考虑 $WX$ 与空间误差 |

因此：

$$
\boxed{
SDEM
=
SLX
+
SEM
}
$$

---

## 17. 关于原文 SDEM 公式需要注意

有些论文可能写成：

$$
Y
=
\rho WY
+
X\beta
+
WX\theta
+
\varepsilon
$$

同时：

$$
\varepsilon
=
\lambda W\varepsilon+\mu
$$

这里同时包含：

$$
WY
$$

$$
WX
$$

和：

$$
W\varepsilon
$$

但严格按照常见的空间计量模型定义：

> 标准 SDEM 通常不包含 $\rho WY$。

标准形式更常见的是：

$$
Y=X\beta+WX\theta+\varepsilon
$$

$$
\varepsilon=\lambda W\varepsilon+\mu
$$

因此在实际论文中应根据：

- 使用的软件
- 模型定义
- 参考文献

明确模型形式。

---

# AIC

## 18. Akaike Information Criterion

AIC 全称：

> **Akaike Information Criterion**
>
> 赤池信息准则

用于比较不同模型的相对拟合表现。

基本公式：

$$
\boxed{
AIC
=
2k
-
2\ln(L)
}
$$

其中：

- $k$：模型需要估计的参数数量
- $L$：Maximum Likelihood
- $\ln(L)$：Log-Likelihood

---

## 19. AIC 的核心思想

AIC 同时考虑两个问题：

### 1. 模型拟合效果

模型拟合越好：

$$
L\uparrow
$$

通常：

$$
-2\ln(L)\downarrow
$$

---

### 2. 模型复杂程度

参数越多：

$$
k\uparrow
$$

那么：

$$
2k\uparrow
$$

模型受到更大的复杂度惩罚。

因此：

$$
\boxed{
AIC
=
拟合效果
+
复杂度惩罚
}
$$

核心思想：

> **模型不能只追求拟合得好，还要避免为了提高拟合效果而加入过多参数。**

---

## 20. AIC 如何判断

最重要的规则：

$$
\boxed{
AIC越小越好
}
$$

例如：

| Model |  AIC |
| ----- | ---: |
| OLS   | 1250 |
| SEM   | 1170 |
| SDM   | 1145 |
| SDEM  | 1120 |

则：

$$
AIC_{SDEM}=1120
$$

最小。

说明在这些候选模型之间：

> SDEM 在模型拟合与模型复杂度之间取得了更好的平衡。

---

## 21. $\Delta AIC$

AIC 本身没有绝对的“好”或“坏”。

通常需要比较：

$$
\boxed{
\Delta AIC_i
=
AIC_i-AIC_{min}
}
$$

例如：

| Model |  AIC | $\Delta AIC$ |
| ----- | ---: | -----------: |
| SDEM  | 1120 |            0 |
| SDM   | 1122 |            2 |
| SEM   | 1135 |           15 |

一般可以粗略理解为：

- $\Delta AIC<2$：模型支持程度非常接近
- $\Delta AIC\approx4-7$：差异逐渐明显
- $\Delta AIC>10$：AIC 较高模型得到的数据支持明显较弱

因此：

$$
\boxed{
不能只看AIC绝对值
}
$$

而应该：

$$
\boxed{
在同一数据和同一因变量下比较不同模型
}
$$

---

# Jarque-Bera Test

## 22. Jarque-Bera 检验是什么

Jarque-Bera Test，简称：

$$
JB
$$

用于检验：

> **模型残差是否显著偏离正态分布。**

它主要根据残差的：

- Skewness（偏度）
- Kurtosis（峰度）

进行判断。

---

## 23. JB 检验的假设

原假设：

$$
\boxed{
H_0:
Residuals\ follow\ a\ normal\ distribution
}
$$

即：

> 残差服从正态分布。

备择假设：

$$
\boxed{
H_1:
Residuals\ are\ non-normal
}
$$

即：

> 残差显著偏离正态分布。

---

## 24. JB 检验如何判断

如果：

$$
p>0.05
$$

则：

> 不能拒绝 $H_0$，没有足够证据认为残差显著偏离正态分布。

如果：

$$
p<0.05
$$

则：

> 拒绝 $H_0$，残差存在显著非正态性。

因此：

$$
\boxed{
JB显著
\Rightarrow
Non-normality
}
$$

---

## 25. 为什么要检查正态性

理想情况下，OLS 残差应大致表现为：

- 大多数残差接近 0
- 极大正残差较少
- 极大负残差较少
- 分布大致对称

即近似：

$$
\varepsilon\sim N(0,\sigma^2)
$$

如果残差严重偏离正态分布：

> 某些经典的显著性检验与统计推断需要更加谨慎。

---

# Heteroskedasticity

## 26. 什么是异方差

OLS 一个经典假设是：

$$
\boxed{
Var(\varepsilon_i)=\sigma^2
}
$$

即：

> 所有观测值的误差波动程度大致相同。

这称为：

> **Homoskedasticity（同方差）**

---

如果不同观测的误差方差不同：

$$
Var(\varepsilon_i)
\neq
Var(\varepsilon_j)
$$

则称为：

> **Heteroskedasticity（异方差）**

---

## 27. 异方差的直观例子

例如：

| 地区 | 实际值 | 预测值 | Residual |
| ---- | -----: | -----: | -------: |
| A    |     20 |     18 |       +2 |
| B    |     25 |     22 |       +3 |
| C    |    200 |    150 |      +50 |
| D    |    250 |    180 |      +70 |

可以发现：

> 当观测规模增大时，残差的波动也明显增大。

即：

$$
X\uparrow
$$

同时：

$$
Var(\varepsilon)\uparrow
$$

这就是典型的异方差。

---

# Breusch-Pagan Test

## 28. Breusch-Pagan 检验是什么

Breusch-Pagan Test，简称：

$$
BP
$$

用于检验：

> **模型残差的方差是否随着解释变量发生系统变化。**

换句话说：

> 不同观测值的误差波动是不是一样大。

---

## 29. BP 检验的假设

原假设：

$$
\boxed{
H_0:
Homoskedasticity
}
$$

即：

$$
Var(\varepsilon_i)=\sigma^2
$$

备择假设：

$$
\boxed{
H_1:
Heteroskedasticity
}
$$

---

## 30. BP 检验如何判断

如果：

$$
p>0.05
$$

则：

> 不能拒绝同方差假设，没有显著异方差证据。

如果：

$$
p<0.05
$$

则：

> 拒绝同方差假设，存在显著异方差。

因此：

$$
\boxed{
BP显著
\Rightarrow
Heteroskedasticity
}
$$

---

# Koenker-Bassett Test

## 31. Koenker-Bassett 检验是什么

Koenker-Bassett Test，简称：

$$
KB
$$

同样主要用于检验：

> **Heteroskedasticity（异方差）**

可以把它理解为：

> **对非正态误差更加稳健的异方差检验。**

它与 Breusch-Pagan Test 检查的问题相近，但对误差非正态等情况更加稳健。

---

## 32. KB 检验的假设

原假设：

$$
\boxed{
H_0:
Homoskedasticity
}
$$

即：

> 残差方差恒定。

备择假设：

$$
\boxed{
H_1:
Heteroskedasticity
}
$$

即：

> 残差方差不是恒定的。

---

## 33. KB 检验如何判断

如果：

$$
p>0.05
$$

则：

> 没有显著异方差证据。

如果：

$$
p<0.05
$$

则：

> 存在显著异方差。

因此：

$$
\boxed{
KB显著
\Rightarrow
Heteroskedasticity
}
$$

---

## 34. BP 和 KB 的区别

二者都用于判断异方差。

### Breusch-Pagan

$$
\boxed{
BP
=
传统异方差检验
}
$$

### Koenker-Bassett

$$
\boxed{
KB
=
对非正态性更加稳健的异方差检验
}
$$

因此如果：

$$
JB
$$

显示残差明显非正态，

那么：

$$
KB
$$

的结果通常尤其值得关注。

---

# 三个检验放在一起理解

## 35. JB、BP、KB 分别检查什么

| Test            | 检查内容   | $H_0$            | 显著意味着   |
| --------------- | ---------- | ---------------- | ------------ |
| Jarque-Bera     | 正态性     | 残差服从正态分布 | 存在非正态性 |
| Breusch-Pagan   | 异方差     | 残差同方差       | 存在异方差   |
| Koenker-Bassett | 稳健异方差 | 残差同方差       | 存在异方差   |

---

## 36. 一个完整例子

假设结果为：

| Test            | p-value |
| --------------- | ------: |
| Jarque-Bera     |   0.001 |
| Breusch-Pagan   |   0.003 |
| Koenker-Bassett |   0.008 |

### Jarque-Bera

$$
0.001<0.05
$$

因此：

$$
\boxed{
Residuals\ are\ non-normal
}
$$

即：

> 残差显著偏离正态分布。

---

### Breusch-Pagan

$$
0.003<0.05
$$

因此：

$$
\boxed{
Heteroskedasticity
}
$$

即：

> 存在显著异方差。

---

### Koenker-Bassett

$$
0.008<0.05
$$

因此：

$$
\boxed{
Robust\ evidence\ of\ heteroskedasticity
}
$$

即：

> 在较稳健的检验下，异方差仍然显著。

---

## 37. 这三个检验能不能说明存在空间自相关

不能。

需要特别区分：

$$
\boxed{
Non-normality
\neq
Spatial\ Autocorrelation
}
$$

$$
\boxed{
Heteroskedasticity
\neq
Spatial\ Autocorrelation
}
$$

Jarque-Bera 检查：

> 残差的分布形态。

BP / KB 检查：

> 残差方差是否恒定。

而空间自相关需要检查：

> 相邻地区的残差是否存在空间关联。

通常使用：

- Moran's I
- LM-Lag
- LM-Error
- Robust LM-Lag
- Robust LM-Error

---

# 从 OLS 到空间模型的诊断逻辑

## 38. OLS 模型诊断可以理解成一次“体检”

### 1. Jarque-Bera

回答：

> **残差像不像正态分布？**

$$
\boxed{
JB
\rightarrow
Normality
}
$$

---

### 2. Breusch-Pagan

回答：

> **不同地区的误差波动是不是一样大？**

$$
\boxed{
BP
\rightarrow
Heteroskedasticity
}
$$

---

### 3. Koenker-Bassett

回答：

> **在对非正态性更加稳健的情况下，是否仍然存在异方差？**

$$
\boxed{
KB
\rightarrow
Robust\ Heteroskedasticity
}
$$

---

### 4. Moran's I

回答：

> **残差是不是在空间上扎堆？**

$$
\boxed{
Moran's\ I
\rightarrow
Spatial\ Autocorrelation
}
$$

---

### 5. LM Tests

回答：

> **这种空间依赖更接近因变量空间滞后还是空间误差？**

例如：

$$
LM-Lag
$$

和：

$$
LM-Error
$$

---

### 6. Spatial Model

根据空间诊断进一步比较：

- SLM
- SEM
- SDM
- SDEM

并结合：

- Log-Likelihood
- AIC
- BIC / SC
- $R^2$
- residual diagnostics

进行模型比较。

---

# 最终核心总结

## 39. SDEM

$$
\boxed{
Y
=
X\beta
+
WX\theta
+
\varepsilon
}
$$

$$
\boxed{
\varepsilon
=
\lambda W\varepsilon+\mu
}
$$

回答两个空间问题：

$$
\boxed{
WX:
邻居的X会不会影响我？
}
$$

$$
\boxed{
W\varepsilon:
没有观测到的空间因素是否仍然使残差相关？
}
$$

---

## 40. AIC

$$
\boxed{
AIC=2k-2\ln(L)
}
$$

回答：

> **模型在拟合效果和复杂程度之间是否取得较好的平衡？**

判断：

$$
\boxed{
AIC越小越好
}
$$

但必须：

> 在同一数据和同一因变量的候选模型之间进行比较。

---

## 41. Jarque-Bera

回答：

> **残差是否显著偏离正态分布？**

$$
\boxed{
p<0.05
\Rightarrow
Non-normality
}
$$

---

## 42. Breusch-Pagan

回答：

> **残差是否存在异方差？**

$$
\boxed{
p<0.05
\Rightarrow
Heteroskedasticity
}
$$

---

## 43. Koenker-Bassett

回答：

> **在更加稳健的条件下，残差是否仍存在异方差？**

$$
\boxed{
p<0.05
\Rightarrow
Heteroskedasticity
}
$$

---

# 一句话速记

$$
\boxed{
JB
=
看残差的形状
}
$$

$$
\boxed{
BP
=
看残差的大小是否稳定
}
$$

$$
\boxed{
KB
=
更稳健地看残差大小是否稳定
}
$$

$$
\boxed{
Moran's\ I
=
看残差在空间上是否扎堆
}
$$

$$
\boxed{
SDEM
=
本地X
+
邻居X
+
空间相关误差
}
$$

$$
\boxed{
AIC
=
拟合效果
+
复杂度惩罚
}
$$

---

# 最终逻辑链

$$
\boxed{
OLS
}
$$

↓

检查残差是否正态：

$$
\boxed{
Jarque-Bera
}
$$

↓

检查是否异方差：

$$
\boxed{
Breusch-Pagan
+
Koenker-Bassett
}
$$

↓

检查残差是否存在空间自相关：

$$
\boxed{
Moran's\ I
}
$$

↓

判断空间依赖类型：

$$
\boxed{
LM-Lag
+
LM-Error
}
$$

↓

建立并比较空间模型：

$$
\boxed{
SLM,\ SEM,\ SDM,\ SDEM
}
$$

↓

利用：

$$
\boxed{
AIC,\ BIC,\ LogLikelihood,\ R^2
}
$$

进行模型比较

↓

最终选择适合数据空间结构的模型。