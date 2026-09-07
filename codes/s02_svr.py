# -*- coding: utf-8 -*-
# 依赖
import argparse
import os
import json
from shap_visualization import save_shap_visualization
from data_split import load_groups, split_samples
from matplotlib.figure import Figure
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold, GroupKFold, ParameterGrid
from sklearn.model_selection import _search
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR


# 参数
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = None
GROUPS_PATH = None
MODEL_GROUP = 's02_svr'
N_JOBS = min(4, os.cpu_count() or 1)
OUTPUT_DIR = None

class RegressionAnalysis:
    # 初始化
    def __init__(self, data_path: str, output_dir: str = None, random_state: int = 42, groups_path: str = None):
        self.results = []
        self.models = {}
        self.search_results = {}
        if data_path is None:
            raise ValueError('请通过--data传入数据文件绝对路径')
        self.groups_path = groups_path
        self.groups = None
        self.data_path = Path(data_path)
        if not self.data_path.is_absolute():
            raise ValueError('DATA_PATH须为绝对路径')
        self.random_state = random_state
        self.cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = ROOT / 'results' / self.data_path.stem / MODEL_GROUP
            self.output_dir.mkdir(parents=True, exist_ok=True)


    # 评估与绘图
    def _evaluate_model(self, est, model_name, X_train, X_test, y_train, y_test):
        y_tr_pred, y_te_pred = est.predict(X_train), est.predict(X_test)
        train_r2, test_r2 = r2_score(y_train, y_tr_pred), r2_score(y_test, y_te_pred)
        rec = {
            'Model': model_name,
            'CV_RMSE': self.search_results[model_name]['cv_rmse'],
            'Train_R2': train_r2,
            'Train_RMSE': float(np.sqrt(mean_squared_error(y_train, y_tr_pred))),
            'Train_MAE': mean_absolute_error(y_train, y_tr_pred),
            'Test_R2': test_r2,
            'Test_RMSE': float(np.sqrt(mean_squared_error(y_test, y_te_pred))),
            'Test_MAE': mean_absolute_error(y_test, y_te_pred),
            'Overfitting_Score': train_r2-test_r2,
        }
        self.results.append(rec)
        self.models[model_name] = est
        pd.DataFrame({'row_index': y_test.index, 'Actual': y_test.to_numpy(),
                      'Predicted': y_te_pred}).to_csv(
            self.output_dir / f'{model_name}_predictions.csv', index=False, encoding='utf-8-sig')
        fig = Figure(figsize=(10, 6))
        ax = fig.subplots()
        ax.scatter(y_test, y_te_pred, alpha=0.3)
        low = float(min(np.min(y_test), np.min(y_te_pred)))
        high = float(max(np.max(y_test), np.max(y_te_pred)))
        ax.plot([low, high], [low, high], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{model_name}: Actual vs Predicted')
        fig.tight_layout()
        fig.savefig(self.output_dir / f'{model_name}_actual_predicted.png', dpi=200)
        save_shap_visualization(est, X_train, X_test, model_name, self.output_dir, self.random_state)
        print(f"{model_name} | CV RMSE: {rec['CV_RMSE']:.4f} | Test R2: {test_r2:.4f} | Test RMSE: {rec['Test_RMSE']:.4f}")
        return rec

    # 数据读取与划分
    def load_and_split(self):
        if not self.data_path.is_file():
            raise FileNotFoundError(f"数据文件未找到: {self.data_path}")
        suffix = self.data_path.suffix.lower()
        if suffix == '.csv':
            data = pd.read_csv(self.data_path, encoding='utf-8-sig')
        elif suffix in {'.xlsx', '.xls'}:
            data = pd.read_excel(self.data_path)
        else:
            raise ValueError('输入文件须为CSV、XLSX或XLS')
        if data.shape[1] < 2:
            raise ValueError('第1列为Y，第2列之后为X，至少需要两列')
        try:
            data = data.apply(pd.to_numeric, errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError('Y和X须为数值，不应包含城市名、说明或其他文本列') from exc
        data = data.replace([np.inf, -np.inf], np.nan)
        groups = load_groups(self.groups_path, len(data))
        valid = data.iloc[:, 0].notna()
        if not valid.all():
            print(f'排除Y缺失或无穷值记录: {(~valid).sum()}行')
        data = data.loc[valid]
        self.groups = groups.loc[data.index] if groups is not None else None
        y, X = data.iloc[:, 0], data.iloc[:, 1:]
        if len(data) < 15:
            raise ValueError('有效样本不足以进行7:3划分和3折交叉验证')
        if X.isna().all().any():
            raise ValueError(f'存在整列缺失的X: {X.columns[X.isna().all()].tolist()}')
        if y.nunique() < 2:
            raise ValueError('Y为常数，无法使用R²评估')
        self.cv = (GroupKFold(n_splits=3, shuffle=True, random_state=self.random_state)
                   if self.groups is not None else KFold(n_splits=3, shuffle=True, random_state=self.random_state))
        X_train, X_test, y_train, y_test = split_samples(
            X, y, self.groups, self.cv, self.random_state, self.output_dir)
        if X_train.isna().all().any():
            raise ValueError('训练集中有X整列缺失，请检查输入数据')
        if not X_train.nunique().gt(1).any():
            raise ValueError('训练集没有可变化的特征')
        print(f'样本: {len(data)}，训练: {len(X_train)}，测试: {len(X_test)}，X: {X.shape[1]}')
        return X_train, X_test, y_train, y_test

    # 预处理
    def _preprocess_steps(self, scale: bool):
        steps = [
            ('imputer', SimpleImputer(strategy="median")),
            ('var', VarianceThreshold(threshold=0.0)),
        ]
        if scale:
            steps.append(('scaler', StandardScaler()))
        return steps

    # 网格搜索
    def _fit_grid(self, name, reg, param_grid, X_train, y_train, scale: bool):
        pipe = Pipeline(self._preprocess_steps(scale=scale) + [('reg', reg)])
        search = GridSearchCV(
            estimator=pipe, param_grid=param_grid,
            scoring='neg_mean_squared_error', cv=self.cv,
            n_jobs=N_JOBS, verbose=0, error_score='raise', refit=True
        )
        total = len(ParameterGrid(search.param_grid)) * self.cv.get_n_splits(X_train, y_train)
        original_parallel = _search.Parallel
        with tqdm(total=total + 1, desc=f'{name} CV', unit='fit', dynamic_ncols=True) as progress:
            class ProgressParallel(original_parallel):
                def print_progress(self):
                    progress.update(max(0, min(self.n_completed_tasks, total) - progress.n))
                    if self.n_completed_tasks == total:
                        progress.set_description(f'{name} refit')

            _search.Parallel = ProgressParallel
            try:
                groups = self.groups.loc[X_train.index] if self.groups is not None else None
                search.fit(X_train, y_train, groups=groups)
                progress.update(1)
                progress.set_description(name)
            finally:
                _search.Parallel = original_parallel
        best_params = {key.removeprefix('reg__'): value for key, value in search.best_params_.items()}
        self.search_results[name] = {
            'best_params': best_params,
            'cv_mse': float(-search.best_score_),
            'cv_rmse': float(np.sqrt(-search.best_score_)),
            'cv_folds': self.cv.n_splits,
            'scoring': search.scoring,
            'split_unit': 'group' if self.groups is not None else 'row',
            'random_state': self.random_state,
        }
        (self.output_dir / f'{name}_best_params.json').write_text(
            json.dumps(self.search_results[name], ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{name} Best parameters: {best_params}')
        return name, search.best_estimator_

    # SVR
    def train_svr(self, X_train, y_train):
        svr = SVR()
        param_grid = {
            'reg__kernel': ['rbf', 'linear'],
            'reg__C': [0.1, 1, 10, 100, 300],
            'reg__epsilon': [0.01, 0.05, 0.1, 0.5, 1.0],
            'reg__gamma': ['scale', 'auto'],  # 对 linear 会被忽略
        }
        return self._fit_grid("SVR", svr, param_grid, X_train, y_train, scale=True)

    # 训练流程
    def run_analysis(self):
        self.results.clear()
        self.models.clear()
        self.search_results.clear()
        X_train, X_test, y_train, y_test = self.load_and_split()
        trainers = [self.train_svr]
        for trainer in trainers:
            name, est = trainer(X_train, y_train)
            self._evaluate_model(est, name, X_train, X_test, y_train, y_test)
        out_file = self.save_results_to_excel()
        print(f'Results: {out_file}')

    # 结果保存
    def save_results_to_excel(self):
        if not self.results:
            raise RuntimeError("无可保存的结果，请先运行模型训练。")

        df = pd.DataFrame(self.results).sort_values('CV_RMSE', ascending=True)

        metric_cols = ['Train_R2', 'Train_RMSE', 'Train_MAE', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'Overfitting_Score']
        for c in metric_cols:
            df[c] = df[c].astype(float)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f'SVR_Model_Results_{timestamp}.xlsx'

        with pd.ExcelWriter(str(filename)) as writer:
            df.to_excel(writer, sheet_name='模型性能汇总', index=False)

            analysis = pd.DataFrame({
                '指标': ['模型', '测试R2', '平均过拟合程度'],
                '值': [
                    df.iloc[0]['Model'],
                    f"{df.iloc[0]['Test_R2']:.3f}",
                    f"{df['Overfitting_Score'].mean():.3f}",
                ]
            })
            analysis.to_excel(writer, sheet_name='性能分析', index=False)

        return str(filename)


# 运行入口
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DATA_PATH, required=DATA_PATH is None, help='数据文件绝对路径')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR, help='结果文件夹')
    parser.add_argument('--check-only', action='store_true', help='只检查数据，不训练')
    parser.add_argument('--groups', type=Path, default=GROUPS_PATH, help='分组CSV绝对路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    analyzer = RegressionAnalysis(data_path=args.data, output_dir=args.output, random_state=args.seed, groups_path=args.groups)
    if args.check_only:
        analyzer.load_and_split()
    else:
        analyzer.run_analysis()


if __name__ == '__main__':
    main()
