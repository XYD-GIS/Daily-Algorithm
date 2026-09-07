# 依赖
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from tqdm.auto import tqdm


# 参数
BACKGROUND_SIZE = 100
PERMUTATIONS = 10
BATCH_SIZE = 25
FEATURE_LABELS = {}


# SHAP计算
def explain_model(est, X_train, X_test, model_name, random_state=42):
    prep, reg = est[:-1], est.named_steps['reg']
    train = np.asarray(prep.transform(X_train), dtype=float)
    test = np.asarray(prep.transform(X_test), dtype=float)
    names = list(prep.get_feature_names_out(X_train.columns))
    background = shap.sample(train, min(BACKGROUND_SIZE, len(train)), random_state=random_state)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    fallback = None

    if hasattr(reg, 'get_scale_and_bias') and hasattr(reg, 'get_feature_importance'):
        explainer = shap.PermutationExplainer(reg.predict, masker, seed=random_state)
        method = 'permutation_interventional'
        fallback = 'CatBoost: use predict directly to preserve model scale and bias'
    elif hasattr(reg, 'coef_'):
        explainer = shap.LinearExplainer(reg, masker)
        method = 'linear_interventional'
    else:
        try:
            explainer = shap.TreeExplainer(reg, data=background, feature_perturbation='interventional')
            method = 'tree_interventional'
        except (shap.utils._exceptions.InvalidModelError, ValueError, NotImplementedError) as exc:
            fallback = f'{type(exc).__name__}: {exc}'
            explainer = shap.PermutationExplainer(reg.predict, masker, seed=random_state)
            method = 'permutation_interventional'

    values, bases = [], []
    with tqdm(total=len(test), desc=f'{model_name} SHAP', unit='sample', dynamic_ncols=True) as progress:
        for start in range(0, len(test), BATCH_SIZE):
            batch = test[start:start+BATCH_SIZE]
            if method == 'permutation_interventional':
                result = explainer(batch, max_evals=(2*test.shape[1]+1)*PERMUTATIONS, silent=True)
            else:
                result = explainer(batch)
            values.append(np.asarray(result.values, dtype=float))
            bases.append(np.broadcast_to(np.asarray(result.base_values).reshape(-1), (len(batch),)))
            progress.update(len(batch))
    values, bases = np.concatenate(values), np.concatenate(bases)
    predictions = np.asarray(est.predict(X_test))
    if values.shape != test.shape or not np.isfinite(values).all():
        raise ValueError(f'{model_name}: SHAP维度或数值异常')
    np.testing.assert_allclose(bases+values.sum(axis=1), predictions, rtol=1e-4, atol=1e-6)
    colors = np.asarray(est.named_steps['imputer'].transform(X_test))
    colors = colors[:, est.named_steps['var'].get_support()]
    metadata = {
        'model': model_name, 'method': method, 'background_source': 'training_set',
        'background_size': len(background), 'explained_source': 'test_set',
        'explained_size': len(test), 'random_state': random_state,
        'permutations': PERMUTATIONS if method == 'permutation_interventional' else None,
        'fallback_reason': fallback, 'features': names,
        'excluded_features': [x for x in X_train.columns if x not in names],
        'max_additivity_error': float(np.max(np.abs(bases+values.sum(axis=1)-predictions))),
    }
    return values, bases, colors, names, metadata


# 组合图
def plot_shap_combined(values, colors, names, model_name, output_dir, random_state=42):
    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(-importance, kind='stable')
    labels = [FEATURE_LABELS.get(x, x) for x in names]
    explanation = shap.Explanation(values=values, data=colors, feature_names=labels)
    style = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
             'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 11,
             'axes.linewidth': 1.1}
    with mpl.rc_context(style):
        fig, ax = plt.subplots(figsize=(8.2, max(4.6, 0.52*len(names)+1.8)))
        random_state_before = np.random.get_state()
        try:
            np.random.seed(random_state)
            shap.plots.beeswarm(explanation, max_display=len(names), order=order,
                color=plt.get_cmap('RdYlBu_r'), ax=ax, plot_size=None, show=False,
                alpha=0.85, s=15, color_bar=True, group_remaining_features=False)
            ax.set_xlabel('SHAP value')
            ax.set_ylabel('')
            ax.set_title(model_name, pad=48, fontsize=13)
            limit = max(float(np.max(np.abs(values)))*1.08, 1e-8)
            ax.set_xlim(-limit, limit)
            ax.set_zorder(2)
            ax.patch.set_visible(False)

            bar_ax = ax.twiny()
            bar_ax.set_zorder(1)
            positions = np.arange(len(names))[::-1]
            bar_colors = plt.get_cmap('RdYlBu_r')(np.linspace(0.88, 0.12, len(names)))
            bar_ax.barh(positions, importance[order], height=0.62,
                        color=bar_colors, alpha=0.35, edgecolor='#333333', linewidth=1.0)
            bar_ax.set_xlim(0, max(float(importance.max())*1.12, 1e-8))
            bar_ax.set_xlabel('Mean |SHAP value|', labelpad=9)
            bar_ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
            bar_ax.tick_params(axis='y', left=False, right=False, labelleft=False)
            bar_ax.set_ylim(ax.get_ylim())
            bar_ax.spines['bottom'].set_visible(False)
            for suffix in ('svg', 'png'):
                fig.savefig(output_dir / f'{model_name}_shap_combined.{suffix}',
                            dpi=300, bbox_inches='tight', facecolor='white')
        finally:
            np.random.set_state(random_state_before)
            plt.close(fig)
    return importance, order


# 输出
def save_shap_visualization(est, X_train, X_test, model_name, output_dir, random_state=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values, bases, colors, names, metadata = explain_model(
        est, X_train, X_test, model_name, random_state)
    importance, order = plot_shap_combined(values, colors, names, model_name, output_dir, random_state)
    pd.DataFrame({'feature': np.asarray(names)[order], 'mean_abs_shap': importance[order]}).to_csv(
        output_dir / f'{model_name}_shap_importance.csv', index=False, encoding='utf-8-sig')
    table = pd.DataFrame(values, columns=names)
    table.insert(0, 'base_value', bases)
    table.insert(0, 'row_index', X_test.index.to_numpy())
    table.to_csv(output_dir / f'{model_name}_shap_values.csv', index=False, encoding='utf-8-sig')
    (output_dir / f'{model_name}_shap_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
