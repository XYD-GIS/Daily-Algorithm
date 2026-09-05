# 2026.09.06XGBoost

# 核心思路及优点

将多个简单的模型（通常是决策树）组合在一起，逐步改进模型的预测能力。

- **高效：** 算法速度快，计算资源利用效率高。
- **准确：** 可以处理复杂的数据，生成高质量的预测。
- **灵活：** 可以处理缺失值；类别变量是否可直接处理取决于版本和参数设置，很多场景仍会先做编码。
- **可扩展：** 适用于大规模数据集，可以在分布式环境中运行。

# 理论基础

## 目标函数

XGBoost 的目标函数由两个部分组成：损失函数和正则化项。

目标函数的表达式为：

$$
\mathcal{L}(\theta)
=
\sum_{i=1}^{n} l(y_i,\hat{y}_i)
+
\sum_{k=1}^{K}\Omega(f_k)
$$

其中：

- yᵢ 是真实值
- ŷᵢ 是预测值
- l 是损失函数，通常为均方误差（MSE）或对数损失
- fₖ 是第 k 棵树
- Ω 是正则化项，用于控制模型的复杂度

## 梯度提升

XGBoost 通过逐步添加树来最小化损失函数，每一步都是在前一步的基础上改进预测。

假设我们已经有了 t−1 棵树，第 t 棵树的目标是最小化下述的增量损失函数：

$$
\mathcal{L}^{(t)}
=
\sum_{i=1}^{n}
l\left(
y_i,
\hat{y}_i^{(t-1)}+f_t(x_i)
\right)
+
\Omega(f_t)
$$

其中：

- ŷᵢ⁽ᵗ⁻¹⁾ 是前 t−1 棵树的预测值
- fₜ(xᵢ) 是第 t 棵树的预测值

## 泰勒展开

为了简化目标函数，我们对损失函数进行二阶泰勒展开：

$$
\mathcal{L}^{(t)}
\approx
\sum_{i=1}^{n}
\left[
l\left(
y_i,
\hat{y}_i^{(t-1)}
\right)
+
g_i f_t(x_i)
+
\frac{1}{2}h_i f_t(x_i)^2
\right]
+
\Omega(f_t)
$$

其中：

$$
g_i
=
\frac{
\partial l\left(
y_i,\hat{y}_i^{(t-1)}
\right)
}{
\partial \hat{y}_i^{(t-1)}
}
$$

是一阶导数（梯度）

$$
h_i
=
\frac{
\partial^2 l\left(
y_i,\hat{y}_i^{(t-1)}
\right)
}{
\partial \left(
\hat{y}_i^{(t-1)}
\right)^2
}
$$

是二阶导数（Hessian）

将常数项去掉后，简化为：

$$
\mathcal{L}^{(t)}
\approx
\sum_{i=1}^{n}
\left[
g_i f_t(x_i)
+
\frac{1}{2}h_i f_t(x_i)^2
\right]
+
\Omega(f_t)
$$

## 树模型和正则化

XGBoost 中每棵树的模型可以表示为：

$$
f_t(x)
=
w_{q(x)}
$$

其中：

- q 是叶子节点的索引函数，将输入 x 映射到叶子节点 q(x)
- w 是叶子节点的权重

正则化项定义为：

$$
\Omega(f)
=
\gamma T
+
\frac{1}{2}\lambda
\sum_{j=1}^{T}w_j^2
$$

其中：

- T 是叶子节点的数量
- γ 和 λ 是正则化参数，控制树的复杂度

## 最优化目标

将树模型和正则化项代入简化的目标函数：

$$
\mathcal{L}^{(t)}
=
\sum_{i=1}^{n}
\left[
g_iw_{q(x_i)}
+
\frac{1}{2}h_iw_{q(x_i)}^2
\right]
+
\gamma T
+
\frac{1}{2}\lambda
\sum_{j=1}^{T}w_j^2
$$

对同一个叶子节点的样本求和：

$$
\mathcal{L}^{(t)}
=
\sum_{j=1}^{T}
\left[
G_jw_j
+
\frac{1}{2}(H_j+\lambda)w_j^2
\right]
+
\gamma T
$$

其中：

$$
G_j
=
\sum_{i\in I_j}g_i
$$

是第 j 个叶子节点的梯度之和

$$
H_j
=
\sum_{i\in I_j}h_i
$$

是第 j 个叶子节点的 Hessian 之和

通过对 wⱼ 求导并令其为零，得到最优的叶子节点权重：

$$
w_j^*
=
-\frac{G_j}{H_j+\lambda}
$$

将最优权重代入目标函数，得到最小化后的目标值：

$$
\mathcal{L}^{(t)}
=
-\frac{1}{2}
\sum_{j=1}^{T}
\frac{G_j^2}{H_j+\lambda}
+
\gamma T
$$

## 树的分裂

为了找到最优的分裂点，XGBoost 计算每个候选分裂点的增益：

$$
\mathrm{Gain}
=
\frac{1}{2}
\left(
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right)
-
\gamma
$$

其中：

- Gₗ 和 Hₗ 分别是左子节点的梯度和 Hessian 之和
- Gᵣ 和 Hᵣ 分别是右子节点的梯度和 Hessian 之和

选择增益最大的分裂点进行分裂。

## 算法流程

XGBoost 的整体算法流程：

**1. 初始化：** 初始化预测值 ŷᵢ⁽⁰⁾ 为一个常数值（通常为目标值的平均值）。

**2. 迭代：** 对于每一棵树 t，进行以下步骤：

**a. 计算梯度和 Hessian：** 计算每个样本的梯度 gᵢ 和 Hessian hᵢ。

**b. 构建树：** 逐层分裂节点，计算每个候选分裂点的增益，选择增益最大的分裂点，直至达到预设的叶子节点数或增益小于阈值。

**c. 计算叶子节点权重：** 根据梯度和 Hessian 计算每个叶子节点的权重 wⱼ*。

**3. 更新预测值：** 更新每个样本的预测值：

$$
\hat{y}_i^{(t)}
=
\hat{y}_i^{(t-1)}
+
f_t(x_i)
$$

通过以上步骤，XGBoost 每一轮都在当前模型的基础上补一棵树，用梯度和 Hessian 指导分裂与叶子权重。它和随机森林不同：随机森林的树大多并行、最后投票或平均；XGBoost 的树是按顺序逐步纠错。

# 注意事项

缺失值填充这种需要从数据中计算统计量的操作，应该放在划分训练集和测试集之后进行。否则如果直接用完整数据计算中位数，就相当于提前使用了测试集的信息，容易产生数据泄漏。

**缺失值处理与特征工程**

```python
# 只使用训练集计算中位数
bedrooms_median = X_train['total_bedrooms'].median()

# 训练集和测试集都使用训练集得到的中位数进行填充
X_train['total_bedrooms'] = X_train['total_bedrooms'].fillna(bedrooms_median)
X_test['total_bedrooms'] = X_test['total_bedrooms'].fillna(bedrooms_median)

# 训练集创建特征
X_train['rooms_per_household'] = X_train['total_rooms'] / X_train['households']
X_train['bedrooms_per_room'] = X_train['total_bedrooms'] / X_train['total_rooms']
X_train['population_per_household'] = X_train['population'] / X_train['households']

# 测试集执行相同的特征工程
X_test['rooms_per_household'] = X_test['total_rooms'] / X_test['households']
X_test['bedrooms_per_room'] = X_test['total_bedrooms'] / X_test['total_rooms']
X_test['population_per_household'] = X_test['population'] / X_test['households']

# 将分类特征转换为数值
X_train = pd.get_dummies(X_train, columns=['ocean_proximity'])
X_test = pd.get_dummies(X_test, columns=['ocean_proximity'])

# 保证训练集和测试集的特征列完全一致
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
```

# 参数优化

```python
# 参数网格
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],  #学习率越小，往往需要更多的树
    'n_estimators': [100, 200, 300],	#一共训练多少颗树
    'subsample': [0.8, 1],	#表示每棵树训练时使用多少比例的样本
    'colsample_bytree': [0.8, 1]#表示每棵树使用多少比例的特征
}

# 网格搜索
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# 最优参数
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# 使用最优参数训练模型
xgb_reg_optimized = xgb.XGBRegressor(**best_params)
xgb_reg_optimized.fit(X_train, y_train)

# 预测
y_pred_optimized = xgb_reg_optimized.predict(X_test)

# 均方根误差
rmse_optimized = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
print(f"Optimized RMSE: {rmse_optimized:.4f}")

# 可视化实际值与预测值的关系（优化后）
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_optimized, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Optimized Actual vs Predicted')
plt.show()
```

| 参数               | 数值变小                                     | 数值变大                                       |
| ------------------ | -------------------------------------------- | ---------------------------------------------- |
| `max_depth`        | 树更浅，模型更简单，不容易过拟合             | 树更深，能学更复杂关系，但更容易过拟合         |
| `learning_rate`    | 每棵树修正得少，学习慢，通常需要更多树       | 每棵树修正得多，学习快，但可能学得太猛         |
| `n_estimators`     | 树少，模型简单，可能欠拟合                   | 树多，拟合能力更强，但训练更慢，也可能过拟合   |
| `subsample`        | 每棵树只用部分样本，随机性更强，可抑制过拟合 | 每棵树用更多样本，训练更充分，但随机性降低     |
| `colsample_bytree` | 每棵树只用部分特征，可抑制过拟合             | 每棵树用更多特征，信息更多，但可能更容易过拟合 |

# **使用XGBoost模型的优缺点**

## **优点**：

- **高效性**：XGBoost在计算上非常高效，支持并行和分布式计算，能够处理大型数据集。
- **性能优越**：它通常在结构化数据集上表现出色，具有较高的预测准确率。
- **自动化特征选择**：通过树模型自动选择重要特征，减少了手动特征选择的工作量。
- **防止过拟合**：通过内置的正则化参数（如(\gamma)和(\lambda)）控制模型复杂度，防止过拟合。
- **灵活性**：支持多种目标函数和自定义目标函数，适用于分类和回归问题。
- **处理缺失值**：自动处理缺失值，推测出合适的分支方向。

## **缺点**：

- **参数调优复杂**：需要调整多个超参数以获得最佳性能，调优过程可能较为复杂和耗时。
- **对内存要求较高**：在处理非常大的数据集时，内存占用较高，可能需要大量内存。
- **对数据预处理敏感**：尽管XGBoost自动处理缺失值，但数据标准化、归一化等预处理仍然重要，影响模型性能。
