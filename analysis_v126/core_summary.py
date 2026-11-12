"""按v1.26及R2来源汇总完整核心证据；缺项时不作候选比较。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CURVES = ('C0', 'C1', 'C2', 'C3', 'N1', 'N1_INT')
POINTS = {'N1_ZERO': '0.0', 'C2_ZERO': '0.0', 'N1_ALL': '1.0', 'C2_ALL': '1.0'}
FRACTIONS = ('0.1', '0.2', '0.4', '0.6')
FAMILIES = {'C1': 'ordinary', 'C2': 'ordinary', 'C2_ZERO': 'ordinary', 'C2_ALL': 'ordinary',
            'C3': 'centered', 'N1': 'N1', 'N1_INT': 'N1', 'N1_ZERO': 'N1', 'N1_ALL': 'N1'}
HASH_CACHE = {}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4*1024*1024), b''):
                h.update(block)
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]

def ids_sha(ids):
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def resolve(path, base=ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(base)/path).resolve()

def identity(row):
    seed = row['training_seed']
    return (row['method'], int(row['year']), int(row['role_seed']),
            'NA' if seed == 'NA' else int(seed), str(row['fraction']))

def plan_id(row):
    return identity({**row, 'fraction': row['budget_fraction']})

def validate_completeness(records, expected):
    seen = {}
    for row in records:
        key = identity(row)
        require(key not in seen, '核心身份重复：' + str(key))
        require(key in expected, '存在计划之外的核心身份：' + str(key))
        seen[key] = row
    missing = sorted(set(expected)-set(seen), key=str)
    return {'complete': not missing, 'expected': len(expected), 'observed': len(seen),
            'missing_count': len(missing), 'missing_identities': [list(k) for k in missing]}

def environment_means(rows):
    require(bool(rows), '逐环境结果为空')
    names = [row.get('environment', row.get('env')) for row in rows]
    require(all(isinstance(name, str) for name in names) and len(set(names)) == len(names), '环境身份缺失或重复')
    losses = np.asarray([row['standardized_regret'] for row in rows], dtype=float)
    retentions = np.asarray([row['retention'] for row in rows], dtype=float)
    require(np.isfinite(losses).all() and np.all(losses >= 0), '标准化遗憾非法')
    require(np.isfinite(retentions).all() and np.all((retentions >= 0) & (retentions <= 1)), '保留率非法')
    require(all(isinstance(row['count'], int) and row['count'] > 0 for row in rows), '环境样本数非法')
    tail = np.sort(losses)[::-1]; mass = len(tail)*.2; whole = int(math.floor(mass))
    cvar = float((tail[:whole].sum() + (mass-whole)*tail[whole])/mass)
    return {'standardized_regret': float(losses.mean()), 'retention': float(retentions.mean()),
            'environment_cvar_0.8': cvar, 'environment_quantile_0.8': float(np.quantile(losses, .8))}

def validate_pairing(records):
    shared, budgets = {}, {}
    for row in records:
        key = (row['year'], row['role_seed'])
        envs = tuple(sorted((e.get('environment', e.get('env')), e['count']) for e in row['environment_metrics']))
        signature = (row['audit_ids_sha256'], row['query_count'], envs)
        require(key not in shared or shared[key] == signature, '比较方法混用审计身份、环境或查询池')
        shared[key] = signature
        q, budget, fraction = row['query_count'], row['budget'], row['fraction']
        require(isinstance(q, int) and q > 0 and isinstance(budget, int), '实际预算字段非法')
        expected = int(round(float(fraction)*100))*q//100
        require(budget == expected, '实际整数预算与计划不一致')
        bkey = (*key, fraction)
        signature = (budget, row['support_ids_sha256'])
        require(isinstance(row['support_ids_sha256'], str), '支持ID散列缺失')
        require(bkey not in budgets or budgets[bkey] == signature, '同角色支持预算或名单不一致')
        budgets[bkey] = signature

def actual_auc(values, integer_budgets, query_count):
    x = np.asarray(integer_budgets, dtype=float)/query_count
    y = np.asarray(values, dtype=float)
    require(len(x) == 4 and np.all(np.diff(x) > 0) and np.isfinite(y).all(), '四预算曲线不完整')
    return float(np.sum((y[:-1]+y[1:])*np.diff(x)/2)/(x[-1]-x[0]))

def aggregate_method(records, method):
    """先环境，再同角色网络种子，再角色；ZERO/ALL只保留单点。"""
    point = method in POINTS
    fractions = (POINTS[method],) if point else FRACTIONS
    grouped = defaultdict(dict)
    for row in records:
        if row['method'] == method:
            grouped[(row['year'], row['role_seed'], row['training_seed'])][row['fraction']] = row
    role_values = defaultdict(list)
    for (year, role, seed), curve in grouped.items():
        require(set(curve) == set(fractions), '方法预算曲线不完整')
        values = {f: environment_means(curve[f]['environment_metrics']) for f in fractions}
        primary = values[fractions[0]]['standardized_regret'] if point else actual_auc(
            [values[f]['standardized_regret'] for f in fractions], [curve[f]['budget'] for f in fractions],
            curve[fractions[0]]['query_count'])
        role_values[(year, role)].append({'primary': primary,
                'retention': float(np.mean([values[f]['retention'] for f in fractions])), 'budgets': values})
    annual = defaultdict(list)
    for (year, role), members in sorted(role_values.items()):
        budget_metrics = {f: {metric: float(np.mean([m['budgets'][f][metric] for m in members]))
                             for metric in members[0]['budgets'][f]} for f in fractions}
        annual[year].append({'role_seed': role, 'network_seed_count': len(members),
                            'primary': float(np.mean([m['primary'] for m in members])),
                            'retention': float(np.mean([m['retention'] for m in members])), 'budgets': budget_metrics})
    result = {'kind': 'single_budget' if point else 'four_budget_curve',
              'primary_metric': 'single_budget_standardized_regret' if point else 'actual_budget_normalized_standardized_regret_auc',
              'fraction': fractions[0] if point else None, 'annual': {}}
    for year, roles in sorted(annual.items()):
        result['annual'][year] = {'primary': float(np.mean([r['primary'] for r in roles])),
            'retention': float(np.mean([r['retention'] for r in roles])),
            'budgets': {f: {metric: float(np.mean([r['budgets'][f][metric] for r in roles]))
                           for metric in roles[0]['budgets'][f]} for f in fractions}, 'roles': roles}
    result['macro'] = {key: float(np.mean([r[key] for r in result['annual'].values()])) for key in ('primary', 'retention')}
    return result

def shared_bootstrap(years, draws=2000, seed=12602):
    return np.random.Generator(np.random.PCG64(seed)).integers(0, len(years), size=(draws, len(years)))

def interval(draws):
    return np.quantile(np.asarray(draws, dtype=float), [.025, .975]).tolist()

def compare_years(candidate, baseline, candidate_retention, baseline_retention, indices, denominator_min=1e-8):
    candidate, baseline = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    reduction = baseline-candidate
    retention = np.asarray(candidate_retention)-np.asarray(baseline_retention)
    base_mean = float(baseline.mean())
    relative = float(reduction.mean()/base_mean) if base_mean > denominator_min else None
    bootstrap_base = baseline[indices].mean(axis=1)
    valid = bootstrap_base > denominator_min
    relative_ci = interval(reduction[indices].mean(axis=1)/bootstrap_base) if valid.all() else None
    return {'annual_auc_reduction': reduction.tolist(), 'macro_auc_reduction': float(reduction.mean()),
            'baseline_macro_auc': base_mean, 'relative_auc_reduction': relative,
            'strict_improvement_years': int(np.sum(reduction > 0)),
            'annual_retention_difference': retention.tolist(), 'macro_retention_difference': float(retention.mean()),
            'descriptive_95pct': {'auc_reduction': interval(reduction[indices].mean(axis=1)),
                'retention_difference': interval(retention[indices].mean(axis=1)), 'relative_auc_reduction': relative_ci},
            'relative_interval_invalid_draws': int((~valid).sum()),
            'relative_interpretation': '宏平均差除以基线宏平均；分母不足时不判百分比，不丢弃无效重采样'}

def incremental_gain(n1_zero, n1_auc, c2_zero, c2_auc, indices):
    gn = np.asarray(n1_zero)-np.asarray(n1_auc)
    gc = np.asarray(c2_zero)-np.asarray(c2_auc)
    delta = gn-gc
    return {'formula': '(R_N1_ZERO-AUC_N1)-(R_C2_ZERO-AUC_C2)', 'annual_G_N1': gn.tolist(),
            'annual_G_C2': gc.tolist(), 'annual_delta_G': delta.tolist(), 'macro_G_N1': float(gn.mean()),
            'macro_G_C2': float(gc.mean()), 'macro_delta_G': float(delta.mean()),
            'positive_years': int(np.sum(delta > 0)), 'descriptive_95pct': interval(delta[indices].mean(axis=1)),
            'interpretation': '描述性更新增益差，不是因果效应或田间试验节省证明'}

def support_status(comparison, criteria, gain=None):
    tests = {'years_strict_improvement': comparison['strict_improvement_years'] >= criteria['years_strict_improvement_min'],
             'retention_non_decrease': comparison['macro_retention_difference'] >= criteria['macro_retention_difference_min']}
    if criteria.get('strict_macro_improvement'):
        tests['strict_macro_auc_reduction'] = comparison['macro_auc_reduction'] > 0
    else:
        relative = comparison['relative_auc_reduction']
        tests['relative_auc_reduction'] = None if relative is None else relative >= criteria['relative_auc_reduction_min']
    if gain is not None:
        tests['strict_positive_delta_G'] = gain['macro_delta_G'] > 0
        tests['positive_delta_G_years'] = gain['positive_years'] >= criteria['incremental_gain_positive_years_min']
    status = 'SUPPORTED_EXPLORATORY' if all(v is True for v in tests.values()) else (
        'NOT_DETERMINABLE' if any(v is None for v in tests.values()) and not any(v is False for v in tests.values()) else 'NOT_SUPPORTED')
    return {'status': status, 'criteria': criteria, 'checks': tests}

def summarize_records(records, expected, config):
    completeness = validate_completeness(records, expected)
    if not completeness['complete']:
        return {'state': 'WAITING_EVIDENCE', 'completeness': completeness,
                'scientific_positive_claim_accepted': False, 'note': '核心身份未齐，不计算年度对比或主张阈值'}
    validate_pairing(records)
    methods = {method: aggregate_method(records, method) for method in (*CURVES, *POINTS)}
    years = config['data']['outer_years']
    bootstrap = config['metrics']['bootstrap']
    indices = shared_bootstrap(years, bootstrap['draws'], bootstrap['seed'])
    vector = lambda method, key: np.asarray([methods[method]['annual'][year][key] for year in years])
    for method, result in methods.items():
        result['descriptive_95pct'] = {key: interval(vector(method, key)[indices].mean(axis=1)) for key in ('primary', 'retention')}
    comparisons = {base: compare_years(vector('N1', 'primary'), vector(base, 'primary'), vector('N1', 'retention'),
                   vector(base, 'retention'), indices, config['metrics']['zero_denominator']) for base in ('C2', 'C3', 'N1_INT', 'C0')}
    gain = incremental_gain(vector('N1_ZERO', 'primary'), vector('N1', 'primary'), vector('C2_ZERO', 'primary'), vector('C2', 'primary'), indices)
    claims = {name: support_status(comparisons[spec['comparator']], spec, gain if name == 'V126_H3' else None)
              for name, spec in config['support'].items() if name in ('V126_H1', 'V126_H2', 'V126_H3')}
    practical = {**config['support']['V126_H1'], 'comparator': 'C0'}
    claims['V126_PRACTICAL'] = support_status(comparisons['C0'], practical)
    return {'state': 'CORE_EVIDENCE_SUMMARIZED', 'execution_valid': True, 'completeness': completeness,
            'year_order': years, 'aggregation': config['metrics']['aggregation'], 'methods': methods,
            'N1_comparisons': comparisons, 'incremental_gain': gain, 'claims': claims,
            'bootstrap': {**bootstrap, 'year_indices_sha256': hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                          'same_indices_for_all_methods_and_comparisons': True},
            'evidence_level': '探索性；五个年份整块描述区间，不把网络或角色种子当独立农业样本',
            'SOTA_claim': 'NOT_ASSESSED；核心G2F结果不能单独支持确认性SOTA',
            'history_terminal_states_rewritten': False}


def load_contract(config_path, matrix_path, claims_path, sources_path):
    config, claims, sources = read(config_path), read(claims_path), read(sources_path)
    require(config['plan_version'] == 'v1.26' and config['planned_counts']['core_records'] == 1900, '核心合同版本或数量不符')
    data, metrics = config['data'], config['metrics']
    require(data['outer_years'] == list(range(2020, 2025)) and data['role_seeds'] == [11, 23, 37, 53, 71]
            and data['training_seeds'] == [1009, 2027, 3037] and tuple(str(v) for v in data['budgets']) == FRACTIONS,
            '年份、角色、网络种子或四预算偏离本版合同')
    require(metrics['bootstrap']['draws'] == 2000 and metrics['bootstrap']['seed'] == 12602
            and metrics['zero_denominator'] == 1e-8 and metrics['references_excluded_from_auc'], '统计合同偏离v1.26')
    require({row['id'] for row in claims['claims']} >= {'V126_H1', 'V126_H2', 'V126_H3', 'V126_PRACTICAL'}, '主张映射不完整')
    require(sources['state'] == 'PASS_EXECUTION_SOURCES' and sources['version'] == 'v1.26-R2'
            and sha(matrix_path) == sources['revised_matrix']['sha256'], '未使用已验收R2计划来源')
    with Path(matrix_path).open(encoding='utf-8-sig', newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['stage'] == 'R2']
    expected = {plan_id(row): row for row in rows}
    full = set()
    for method in (*CURVES, *POINTS):
        seeds = ['NA'] if method == 'C0' else data['training_seeds']
        fractions = (POINTS[method],) if method in POINTS else FRACTIONS
        for year in data['outer_years']:
            for role in data['role_seeds']:
                for seed in seeds:
                    for fraction in fractions:
                        full.add((method, year, role, seed, fraction))
    require(len(rows) == len(expected) == len({row['run_id'] for row in rows}) == 1900 and set(expected) == full,
            '核心1900身份并非完整笛卡尔组合，或存在重复/计划外条目')
    return config, expected, sources


def normalize_record(method, year, seed, role_seed, point, audit_hash, source):
    return {'method': method, 'year': year, 'training_seed': seed, 'role_seed': role_seed,
            'fraction': point['fraction'], 'budget': point['budget'], 'query_count': point['query_count'],
            'audit_ids_sha256': audit_hash, 'support_ids_sha256': point['support_ids_sha256'],
            'environment_metrics': point['environment_metrics'], 'source': source}


def load_c0(expected, sources):
    audit_source = sources['all120_M_identity_audit']
    require(sha(audit_source['path']) == audit_source['sha256'], 'M身份审计来源SHA不符')
    audit = read(audit_source['path'])
    require(audit['state'] == 'PASS' and audit['M_count'] == 120, '继承M身份审计尚未通过')
    approved = {}
    for year in audit['years']:
        for group in year['groups']:
            for source in group['M_sources']:
                approved[(year['year'], group['seed'], source['fraction'])] = source
    records, evidence = [], {}
    for key, row in expected.items():
        if key[0] != 'C0':
            continue
        path = resolve(row['source_path'])
        if not path.exists():
            continue
        source = approved[(key[1], key[2], key[4])]
        require(path == resolve(source['source']) and sha(path) == source['sha256'], 'C0来源偏离已验收身份文件')
        saved = read(path)
        require(saved['state'] == 'PASS' and saved['method'] == 'M' and saved['model_spec'] == 'full'
                and saved['year'] == key[1] and saved['seed'] == key[2] and saved['fraction'] == key[4], 'C0身份或完成状态不符')
        if key[1] == 2024:
            require(saved.get('version') == 'v1.26-R2' and saved.get('execution_valid') is True
                    and saved['model_artifact_sha256'] == sources['prepared']['sha256']
                    and saved['source_r2_receipt_sha256'] == sources['input_receipt']['sha256'], '禁止混入旧2024 C0')
        saved['support_ids_sha256'] = ids_sha(saved['selected_ids'])
        records.append(normalize_record('C0', key[1], 'NA', key[2], saved, ids_sha(saved['audit_ids']), str(path)))
        evidence[str(path)] = sha(path)
    return records, evidence


def load_evaluations(index_path, sources, sources_path):
    """索引只列已验收外层评价，不调用评价器，也不打开隐藏标签。"""
    index = read(index_path)
    require(isinstance(index.get('evaluations'), list), '缺少已验收外层评价清单')
    if index['evaluations']:
        require(index['evaluation_code_sha256'] == sha(ROOT/'analysis_v126/outer_evaluate.py'), '评价源码版本不符')
    records, evidence, family_stops = [], {str(Path(index_path).resolve()): sha(index_path)}, {}
    for item in index['evaluations']:
        require(item.get('accepted') is True, '索引包含尚未验收的评价')
        path = resolve(item['path'], Path(index_path).parent)
        manifest_path = resolve(item['manifest_path'], Path(index_path).parent)
        data_dir = resolve(item['data_dir'], Path(index_path).parent)
        require(sha(path) == item['sha256'] and sha(manifest_path) == item['manifest_sha256'], '评价或预测清单SHA不符')
        result, manifest = read(path), read(manifest_path)
        require(result['state'] == 'EXECUTION_VALID' and result['evaluation_only_labels_returned'] is False
                and result['evaluation_code_sha256'] == index['evaluation_code_sha256']
                and result['prediction_manifest_sha256'] == sha(manifest_path), '评价状态或来源关联不符')
        require(manifest['state'] == 'PREDICTIONS_READY' and set(manifest['methods']) == set(result['methods']), '评价方法与预测清单不符')
        for key in ('year', 'training_seed', 'checkpoint_family', 'E_m'):
            require(result[key] == manifest[key], '评价与预测外层身份不符：' + key)
        family = result['checkpoint_family']
        stop = (result['E_m'], manifest['development_result_sha256'])
        require(isinstance(result['E_m'], int) and 0 <= result['E_m'] <= 30, '开发停止轮次非法')
        require(family not in family_stops or family_stops[family] == stop, '同训练分支跨年或网络种子混用开发停止点')
        family_stops[family] = stop
        require(manifest['target_features_sha256'] == sha(data_dir/'target_features.npz')
                and manifest['history_sha256'] == sha(data_dir/'history.npz'), '评价数据与实际导出不符')
        if result['year'] == 2024:
            require(data_dir == resolve(sources['required_2024_data_dir'])
                    and manifest['execution_sources_sha256'] == sha(sources_path), '2024评价未绑定R2新导出')
            for filename, digest in sources['new_export_files'].items():
                # 仅散列核对，无需解析独立隐藏标签。
                require(sha(data_dir/filename) == digest, '2024导出文件SHA不符：' + filename)
        for method, summary in result['methods'].items():
            require(method in (*CURVES, *POINTS) and method != 'C0', '索引混入非核心方法')
            require(FAMILIES[method] == family, '方法使用了错误训练分支')
            require(len(summary['roles']) == 5 and {role['seed'] for role in summary['roles']} == {11, 23, 37, 53, 71}, '外层评价角色缺失或重复')
            if method in POINTS:
                require('score' not in summary and summary['point_fraction'] == POINTS[method], 'ZERO/ALL被错误当作AUC')
            for role in summary['roles']:
                require(role['year'] == result['year'], '评价角色年份不符')
                role_path = data_dir/f"roles/s{role['seed']}/role.json"
                require(sha(role_path) == role['role_sha256'], '角色文件SHA不符')
                role_file = read(role_path)
                audit_hash = ids_sha(role_file['audit_ids'])
                if 'audit_ids_sha256' in role:
                    require(role['audit_ids_sha256'] == audit_hash, '审计ID不一致')
                points = role['budgets'] if method in CURVES else [role]
                for point in points:
                    records.append(normalize_record(method, result['year'], result['training_seed'], role['seed'], point, audit_hash, str(path)))
        evidence[str(path)], evidence[str(manifest_path)] = sha(path), sha(manifest_path)
    return records, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'design_v126/experiment_config.json')
    parser.add_argument('--claims', type=Path, default=ROOT/'design_v126/claim_evidence_map.json')
    parser.add_argument('--matrix', type=Path, default=ROOT/'design_v126/revisions/planned_runs_v1.26-R2.csv')
    parser.add_argument('--r2-sources', type=Path, default=ROOT/'reports/v126/R0_2024_repaired/execution_sources_R2.json')
    args = parser.parse_args()
    config, expected, sources = load_contract(args.config, args.matrix, args.claims, args.r2_sources)
    c0, c0_evidence = load_c0(expected, sources)
    outer, outer_evidence = load_evaluations(args.evaluations_index, sources, args.r2_sources)
    result = summarize_records(c0+outer, expected, config)
    result['source_sha256'] = {**c0_evidence, **outer_evidence, **{str(path.resolve()): sha(path) for path in
                              (args.config, args.claims, args.matrix, args.r2_sources, Path(__file__))}}
    result['input_schema'] = {'evaluations': [{'path': '评价JSON', 'sha256': '评价文件SHA',
        'manifest_path': '预测清单JSON', 'manifest_sha256': '预测清单SHA', 'data_dir': '对应年度导出目录', 'accepted': True}],
        'evaluation_code_sha256': '已验收outer_evaluate.py源码SHA；相对路径相对于索引目录'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'state': result['state'], 'expected': result['completeness']['expected'],
                      'observed': result['completeness']['observed'], 'missing': result['completeness']['missing_count']}))

if __name__ == '__main__':
    main()
