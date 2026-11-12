"""外层独立评价：验证已保存预测，再读取C；零/全观测只报告单点。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis_v126.data import FRACTIONS, ROLE_SEEDS, ids_sha256, load_support, load_target_features, read_json, sha256
from analysis_v126.evaluate import evaluate_role, load_audit, require, unique_strings, upper_cvar
from g2f_design.metrics import ranking_metrics

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    'C1': ('ordinary', 'intercept', FRACTIONS), 'C2': ('ordinary', 'response', FRACTIONS),
    'C3': ('centered', 'response', FRACTIONS), 'N1': ('N1', 'response', FRACTIONS),
    'N1_INT': ('N1', 'intercept', FRACTIONS),
    'C2_ZERO': ('ordinary', 'zero', ('0.0',)), 'N1_ZERO': ('N1', 'zero', ('0.0',)),
    'C2_ALL': ('ordinary', 'response', ('1.0',)), 'N1_ALL': ('N1', 'response', ('1.0',)),
}
BINDING_KEYS = ('year', 'training_seed', 'checkpoint_family', 'E_m', 'checkpoint_sha256',
                'checkpoint_state_sha256', 'development_result_sha256', 'stage_result_sha256',
                'projection_sha256', 'execution_sources_sha256', 'history_sha256',
                'target_features_sha256', 'prediction_code_sha256', 'member_reduction', 'matrix_sha256')


def evaluate_reference_role(data_dir, predictions, seed, fraction, target=None):
    """零观测/全部U沿用同一固定C与原单位指标，不伪造四预算AUC。"""
    require(fraction in ('0.0', '1.0'), '参照仅允许0%或100%')
    data_dir, predictions = Path(data_dir), Path(predictions)
    target = load_target_features(data_dir / 'target_features.npz') if target is None else target
    role_path = data_dir / 'roles' / f's{seed}' / 'role.json'
    role = read_json(role_path)
    year = int(target['outer_year'])
    require(role['year'] == year and role['seed'] == seed, '角色身份错误')
    query, audit_ids = unique_strings(role['query_ids'], '查询'), unique_strings(role['audit_ids'], '审计')
    require(not set(query) & set(audit_ids) and set(query) | set(audit_ids) == set(target['combo_ids']), 'C/U划分错误')
    require(set(role['initial_ids']).issubset(query), '初始支持越权')
    support_path = role_path.parent / f'support_b{fraction}.json'
    support = load_support(support_path, target, role_path)
    require(support['receipt']['fraction'] == fraction, '支持比例身份错误')
    expected = set() if fraction == '0.0' else set(query)
    require(set(support['ids']) == expected, '零观测必须空支持，全观测必须完整U')
    audit_path = data_dir / 'evaluation' / f's{seed}' / 'audit_labels.npz'
    audit = load_audit(audit_path, target, role)
    center, sref = float(audit['center']), float(audit['sref'])
    require(np.isclose(support['receipt']['center'], center, atol=1e-12, rtol=0), '支持与审计中心不一致')
    path = predictions / f'b{fraction}.npz'
    receipt_path = path.with_suffix('.receipt.json')
    receipt = read_json(receipt_path)
    require(receipt['year'] == year and receipt['seed'] == seed and receipt['fraction'] == fraction
            and receipt['budget'] == len(expected), '参照预测预算身份错误')
    require(receipt['role_sha256'] == sha256(role_path) and receipt['support_sha256'] == sha256(support_path)
            and receipt['prediction_sha256'] == sha256(path), '参照预测来源关联错误')
    require(receipt['label_unit'] == 'historical_standardized', '参照预测单位错误')
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {'combo_ids', 'prediction'}, '预测字段错误')
        ids = unique_strings(archive['combo_ids'], '预测')
        prediction = np.asarray(archive['prediction'], dtype=np.float64)
    require(set(ids) == set(audit_ids) and prediction.shape == (len(ids),)
            and np.isfinite(prediction).all(), '预测未完整覆盖C或数值非法')
    lookup = {key: i for i, key in enumerate(ids)}
    raw = center + sref * prediction[[lookup[key] for key in audit['combo_ids']]]
    require(np.isfinite(raw).all(), '恢复原单位后预测非有限')
    environments = []
    for env in np.unique(audit['env']):
        mask = audit['env'] == env
        y, p = audit['y_raw'][mask], raw[mask]
        metrics = ranking_metrics(y, p, audit['hybrid_ids'][mask].tolist())
        metrics.update(environment=str(target['env_names'][int(env)]), count=int(mask.sum()),
                       standardized_regret=metrics['regret'] / sref,
                       centered_rmse=float(np.sqrt(np.mean(((y - y.mean()) - (p - p.mean())) ** 2))))
        environments.append(metrics)
    regrets = [e['standardized_regret'] for e in environments]
    correlations = [e['spearman'] for e in environments if e['spearman'] is not None]
    return dict(year=year, seed=seed, fraction=fraction, budget=len(expected), query_count=len(query),
                actual_fraction=len(expected) / len(query), macro_standardized_regret=float(np.mean(regrets)),
                macro_retention=float(np.mean([e['retention'] for e in environments])),
                macro_rmse=float(np.mean([e['rmse'] for e in environments])),
                macro_centered_rmse=float(np.mean([e['centered_rmse'] for e in environments])),
                global_rmse=float(np.sqrt(np.mean((audit['y_raw'] - raw) ** 2))),
                macro_spearman=float(np.mean(correlations)) if correlations else None,
                undefined_spearman_environments=len(environments) - len(correlations),
                **{'environment_cvar_0.8': upper_cvar(regrets)}, environment_metrics=environments,
                role_sha256=sha256(role_path), audit_file_sha256=sha256(audit_path),
                prediction_sha256=receipt['prediction_sha256'], prediction_receipt_sha256=sha256(receipt_path),
                support_sha256=sha256(support_path), support_ids_sha256=ids_sha256(support['ids'].tolist()),
                sref=sref, evaluation_only_labels_returned=False)


def validate_manifest(data_dir, manifest_path, r2_sources=None, matrix_path=None):
    data_dir, manifest_path = Path(data_dir).resolve(), Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    require(manifest['state'] == 'PREDICTIONS_READY', '预测尚未完成')
    year, seed = manifest['year'], manifest['training_seed']
    require(year in range(2020, 2025) and seed in (1009, 2027, 3037), '外层身份未获计划支持')
    methods = manifest['methods']
    require(methods and len(methods) == len(set(methods)) and all(m in METHODS for m in methods), '方法清单错误')
    binding = {k: manifest[k] for k in BINDING_KEYS}
    require(binding == read_json(manifest_path.parent / 'execution_binding.json'), '执行绑定与预测清单不一致')
    require(binding['member_reduction'] == 'mean_16' and isinstance(binding['E_m'], int)
            and 0 <= binding['E_m'] <= 30, '成员汇总或停止轮次错误')
    require(binding['history_sha256'] == sha256(data_dir / 'history.npz')
            and binding['target_features_sha256'] == sha256(data_dir / 'target_features.npz'), '评价输入与预测输入不同')
    registry = None
    if year == 2024:
        registry_path = Path(r2_sources or ROOT / 'reports/v126/R0_2024_repaired/execution_sources_R2.json')
        registry = read_json(registry_path)
        require(registry['state'] == 'PASS_EXECUTION_SOURCES' and registry['version'] == 'v1.26-R2'
                and registry['year'] == year, '2024来源未通过')
        require(data_dir == Path(registry['required_2024_data_dir']).resolve()
                == Path(registry['new_export_directory']).resolve(), '2024误用旧目录')
        require(binding['execution_sources_sha256'] == sha256(registry_path), '2024来源绑定错误')
        required_matrix = Path(registry['revised_matrix']['path'])
        require(matrix_path is None or Path(matrix_path).resolve() == required_matrix.resolve(), '2024误用旧矩阵')
        matrix_path = required_matrix
        require(sha256(matrix_path) == registry['revised_matrix']['sha256'], '2024矩阵散列错误')
    else:
        require(binding['execution_sources_sha256'] is None, '历史年份来源身份错误')
        matrix_path = Path(matrix_path or ROOT / 'design_v126/planned_runs.csv')
    require(sha256(matrix_path) == binding['matrix_sha256'], '运行矩阵已改变')
    for relative, expected in manifest['model_readable_files_sha256'].items():
        require(sha256(data_dir / relative) == expected, '预测时输入与当前输入不一致')
    with Path(matrix_path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected_records = {}
    for method in methods:
        family, mode, fractions = METHODS[method]
        require(family == binding['checkpoint_family'], '跨训练分支混用预测')
        if family == 'N1':
            require(binding['E_m'] > 0, 'N1第0轮不得进入外层')
        for role in ROLE_SEEDS:
            for fraction in fractions:
                budget_name = {'0.0': '0', '1.0': '1'}.get(fraction, fraction)
                matched = [r for r in rows if r['method'] == method and r['year'] == str(year)
                           and r['training_seed'] == str(seed) and r['role_seed'] == str(role)
                           and r['budget_fraction'] == budget_name and r['checkpoint_family'] == family and r['stage'] == 'R2']
                require(len(matched) == 1, '评价身份未唯一列入计划')
                expected_records[matched[0]['run_id']] = (method, role, fraction, mode)
    records = manifest['records']
    require(manifest['count'] == len(records) == len(expected_records)
            and len({r['run_id'] for r in records}) == len(records)
            and {r['run_id'] for r in records} == set(expected_records), '预测清单有重复或缺失身份')
    for row in records:
        method, role, fraction, mode = expected_records[row['run_id']]
        path = manifest_path.parent / method / f's{role}' / f'b{fraction}.npz'
        require(Path(row['prediction_path']).resolve() == path, '预测路径不对应当前方法身份')
        receipt_path = path.with_suffix('.receipt.json')
        receipt = read_json(receipt_path)
        require(sha256(receipt_path) == row['receipt_sha256'], '预测清单与回执断开')
        require(all(receipt.get(k) == v for k, v in binding.items()), '预测混用检查点或输入')
        require((receipt['run_id'], receipt['method'], receipt['seed'], receipt['fraction'], receipt['mode'])
                == (row['run_id'], method, role, fraction, mode), '预测方法或角色身份错误')
        require(receipt['prediction_sha256'] == sha256(path)
                and receipt['diagnostics_sha256'] == sha256(path.with_suffix('.diagnostics.json')), '预测或诊断散列错误')
    if registry is not None:
        # 全部预测身份、回执、数组和诊断通过后，独立评价进程才核对包含 C 的完整导出。
        for relative, expected in registry['new_export_files'].items():
            require(sha256(data_dir / relative) == expected, '2024导出散列错误：' + relative)
    return manifest


def evaluate_manifest(data_dir, manifest_path, r2_sources=None, matrix_path=None):
    manifest = validate_manifest(data_dir, manifest_path, r2_sources, matrix_path)
    target = load_target_features(Path(data_dir) / 'target_features.npz')
    require(int(target['outer_year']) == manifest['year'], '评价年份与目标文件不同')
    results = {}
    for method in manifest['methods']:
        fractions = METHODS[method][2]
        directory = Path(manifest_path).resolve().parent / method
        if fractions == FRACTIONS:
            roles = [evaluate_role(data_dir, directory / f's{s}', s, target=target) for s in ROLE_SEEDS]
            summary = dict(score=float(np.mean([r['score'] for r in roles])),
                           score_name='five_role_mean_actual_budget_normalized_standardized_regret_AUC')
        else:
            roles = [evaluate_reference_role(data_dir, directory / f's{s}', s, fractions[0], target) for s in ROLE_SEEDS]
            summary = dict(macro_standardized_regret=float(np.mean([r['macro_standardized_regret'] for r in roles])),
                           point_fraction=fractions[0], score_name='single_budget_standardized_regret_no_auc')
        results[method] = {**summary, 'macro_retention': float(np.mean([r['macro_retention'] for r in roles])), 'roles': roles}
    return dict(state='EXECUTION_VALID', year=manifest['year'], training_seed=manifest['training_seed'],
                checkpoint_family=manifest['checkpoint_family'], E_m=manifest['E_m'], methods=results,
                prediction_manifest_sha256=sha256(Path(manifest_path)),
                evaluation_code_sha256=sha256(Path(__file__)), evidence_level='探索性外层评价',
                scientific_positive_claim_accepted=False, evaluation_only_labels_returned=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--r2-sources', type=Path)
    args = parser.parse_args()
    result = evaluate_manifest(args.data_dir, args.manifest, args.r2_sources)
    if args.output.exists():
        require(read_json(args.output) == result, '已有评价不同，禁止覆盖证据')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('state', 'year', 'training_seed', 'checkpoint_family')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
