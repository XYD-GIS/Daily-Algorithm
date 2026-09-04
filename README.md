<div align="center">

# Daily Algorithm

**算法学习笔记 · 从理解原理到掌握方法**

记录算法思想、梳理建模流程，在学习与复盘中积累。

![Markdown](https://img.shields.io/badge/Notes-Markdown-334155?style=flat-square&logo=markdown&logoColor=white)
![Python](https://img.shields.io/badge/Examples-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

[关于仓库](#关于仓库) · [笔记导航](#笔记导航) · [阅读建议](#阅读建议) · [使用说明](#使用说明)

</div>

---

## 关于仓库

这里是我的算法学习记录，用来整理新知识，也方便日后查阅与复习。

目前主要围绕**机器学习中的树模型与集成学习**展开，收录决策树回归、随机森林及随机森林回归笔记。内容包括直观解释、算法流程、超参数调优示例和模型优缺点，示例使用 Python 与 scikit-learn。

每次学习，尝试回答四个问题：

- **为什么有效？** 理解算法的核心思想与建模方式。
- **如何得到结果？** 梳理从训练到预测的主要步骤。
- **哪些参数值得关注？** 结合示例理解参数的作用与调优方法。
- **何时适合使用？** 总结模型的优势、局限与使用场景。

## 笔记导航

| 主题 | 学习重点 | 示例内容 |
| :--- | :--- | :--- |
| [决策树回归](./决策树回归.md) | 特征划分、递归分裂、叶子节点预测、模型优缺点 | `DecisionTreeRegressor` · 网格搜索 · MSE / R² 评估 |
| [随机森林](./随机森林.md) | 随机采样、随机特征选择、集成预测、模型优缺点 | `RandomForestClassifier` · 网格搜索 · 准确率评分 |
| [随机森林回归](./随机森林回归.md) | Bootstrap 采样、多树平均、树之间的差异、模型优缺点 | `RandomForestRegressor` · 网格搜索 · R² 评分 |

## 阅读建议

**决策树回归 → 随机森林 → 随机森林回归**

1. 从[决策树回归](./决策树回归.md)开始，理解一棵树如何划分样本并输出预测。
2. 阅读[随机森林](./随机森林.md)，认识随机性与多模型集成的思路。
3. 结合[随机森林回归](./随机森林回归.md)，理解多棵回归树如何协同完成预测。

复习时，可以围绕“单棵树与多棵树”“分类与回归”“模型复杂度与泛化能力”进行对照，再回到笔记中的参数示例加深理解。

## 使用说明

- **在线阅读**：点击上方索引即可查看对应笔记。
- **本地阅读**：使用支持 Markdown 的编辑器打开本仓库。
- **代码练习**：笔记中的代码是调参示例片段，运行前需补充必要的导入、数据准备及训练集与测试集划分。

如果发现笔记中的疏漏，欢迎通过 [Issues](https://github.com/XYD-GIS/Daily-Algorithm/issues) 交流与指正。

---

<div align="center">

<sub>学懂一个原理，留下一个清晰的记录。</sub>

</div>
