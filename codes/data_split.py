# 依赖
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# 分组读取
def load_groups(path, n_rows):
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('分组文件须为绝对路径')
    data = pd.read_csv(path, dtype={'group':str})
    if 'group' not in data or len(data) != n_rows or data['group'].isna().any():
        raise ValueError('分组CSV须有group列，行数及顺序须与输入数据一致，不允许空分组')
    if 'row_index' in data and not np.array_equal(data.row_index.to_numpy(), np.arange(n_rows)):
        raise ValueError('分组文件row_index须与输入行号0、1、2……一致')
    return data['group']


# 随机划分
def split_samples(X, y, groups, cv, random_state, output_dir):
    if groups is None:
        tr, te = train_test_split(np.arange(len(X)), test_size=0.3, random_state=random_state)
    else:
        if groups.nunique() < 5:
            raise ValueError('至少需要5个分组以进行7:3划分和3折交叉验证')
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
        tr, te = next(splitter.split(X, y, groups))
        if set(groups.iloc[tr]) & set(groups.iloc[te]):
            raise RuntimeError('训练集与测试集存在相同分组')
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    if ytr.nunique() < 2 or yte.nunique() < 2:
        raise ValueError('训练集或测试集Y为常数，无法可靠评估R²')
    train_groups = groups.loc[Xtr.index] if groups is not None else None
    records = pd.DataFrame({'row_index':X.index, 'split':'test', 'cv_fold':0}, index=X.index)
    records.loc[Xtr.index,'split'] = 'train'
    if groups is not None:
        records['group'] = groups
    for fold, (fit, valid) in enumerate(cv.split(Xtr,ytr,train_groups), start=1):
        if train_groups is not None and set(train_groups.iloc[fit]) & set(train_groups.iloc[valid]):
            raise RuntimeError('交叉验证训练折与验证折存在相同分组')
        records.loc[Xtr.index[valid],'cv_fold'] = fold
    if not records.loc[Xtr.index,'cv_fold'].between(1,3).all():
        raise RuntimeError('交叉验证折分配不完整')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    records.to_csv(output_dir/'split_records.csv',index=False,encoding='utf-8-sig')
    summary = {'split_unit':'group' if groups is not None else 'row', 'random_state':random_state,
               'test_fraction':0.3, 'train_rows':len(tr), 'test_rows':len(te), 'cv_folds':cv.n_splits}
    if groups is not None:
        summary.update(train_groups=int(train_groups.nunique()), test_groups=int(groups.iloc[te].nunique()),
                       group_overlap=0)
        print(f'分组: 训练{summary["train_groups"]}，测试{summary["test_groups"]}，重叠0')
    (output_dir/'split_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return Xtr,Xte,ytr,yte
