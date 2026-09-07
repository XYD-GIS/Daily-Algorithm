# 回归模型

输入为CSV或Excel，第一列Y，其余列X，均为数值。通过 `--data` 传入绝对路径，或修改脚本顶部的 `DATA_PATH`。没有预设项目数据路径。

| 脚本 | 模型 |
|---|---|
| s01_linear.py | LinearRegression、ElasticNet |
| s02_svr.py | SVR |
| s03_knn.py | KNN |
| s04_mlp.py | MLP |
| s05_tree_models.py | RandomForest、ExtraTrees、GBDT、XGBoost、LightGBM、CatBoost、AdaBoost |
| s06_rf.py | RandomForest |

```powershell
python s05_tree_models.py --data "D:/data/input.csv"
python s06_rf.py --data "D:/data/input.csv" --check-only
python s05_tree_models.py --data "D:/data/input.csv" --groups "D:/data/groups.csv" --seed 42
```

默认按样本随机7:3划分，训练集内使用3折随机交叉验证。设置 `--groups` 后按组随机7:3划分，训练集内使用3折分组交叉验证，同一组不会跨训练集、测试集或同一折的训练与验证两侧。7:3针对组数，样本行数不一定恰好7:3。固定种子42，可用 `--seed` 修改。

分组CSV独立于Y/X表，须有 `group` 列，与输入每行一一对应。可加 `row_index` 列（从0起），用于核对行序；分组不参与模型训练。输入改变行序时必须同步调整分组表。缺失Y剔除时同步剔除对应分组。

缺失插补、常数筛除和适用模型的X标准化在训练折内完成。所有模型使用GridSearchCV，评分为负均方误差，自动重拟合最佳模型。调参和SHAP计算均有tqdm进度条。

默认输出至本仓库 `results/输入文件名/模型组/`，也可通过 `--output` 指定。文件包括指标Excel、最优参数、预测CSV、实际值与预测值散点图、SHAP蜂群与柱状叠加图（SVG和300dpi PNG）、SHAP值、重要性及解释配置。`split_records.csv` 和 `split_summary.json` 记录划分与验证折。两种划分实验请使用不同输出目录。

SHAP背景取训练集最多100条，解释全部测试集。柱长为平均绝对SHAP值，点位置为带符号SHAP值，颜色表示特征值高低。适用模型使用线性或Tree SHAP，其余采用Permutation SHAP；所有结果核对SHAP加和与预测一致。解释参数在 `shap_visualization.py`，图中直接使用输入列名。

运行环境须安装 `requirements.txt`。迁移时保留六个入口及 `data_split.py`、`shap_visualization.py`。
