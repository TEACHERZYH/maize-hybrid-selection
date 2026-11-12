"""只用合成评价小表验证权重、方向、完整性和统计门槛。"""
import copy
import json
import numpy as np
import pytest
from analysis_v126.core_summary import (
    CURVES, POINTS, FRACTIONS, aggregate_method, actual_auc, compare_years,
    environment_means, identity, incremental_gain, load_c0, sha,
    shared_bootstrap, summarize_records, support_status, validate_completeness,
)

def row(method='N1', year=2020, role=11, seed=1009, fraction='0.1', value=1., q=101):
    return {'method': method, 'year': year, 'role_seed': role, 'training_seed': seed,
            'fraction': fraction, 'query_count': q, 'budget': int(round(float(fraction)*100))*q//100,
            'audit_ids_sha256': f'{year}-{role}',
            'support_ids_sha256': f'{year}-{role}-{fraction}',
            'environment_metrics': [{'environment': 'A', 'count': 1000, 'standardized_regret': value, 'retention': .8},
                                    {'environment': 'B', 'count': 1, 'standardized_regret': value, 'retention': .8}]}

def criteria(strict=False):
    result = {'relative_auc_reduction_min': .05, 'years_strict_improvement_min': 4,
              'macro_retention_difference_min': 0}
    if strict:
        result.update(strict_macro_improvement=True, relative_auc_reduction_min=0,
                      incremental_gain_positive_years_min=4)
    return result

def test_environment_network_role_weights_are_equal():
    records = []
    for role, values in [(11, [1., 3.]), (23, [5., 7.])]:
        for seed, value in zip([1009, 2027], values):
            for fraction in FRACTIONS:
                item = row(role=role, seed=seed, fraction=fraction, value=value)
                item['environment_metrics'][0]['standardized_regret'] = value-1
                item['environment_metrics'][1]['standardized_regret'] = value+1
                records.append(item)
    summary = aggregate_method(records, 'N1')
    assert summary['macro']['primary'] == pytest.approx(4.)
    assert [r['primary'] for r in summary['annual'][2020]['roles']] == pytest.approx([2., 6.])
    assert environment_means(records[0]['environment_metrics'])['standardized_regret'] == 1.

def test_auc_uses_integer_budget_spacing_and_references_are_points():
    assert actual_auc([2, 4, 1, 3], [1, 2, 5, 7], 13) == pytest.approx(14.5/6)
    for method, fraction in POINTS.items():
        result = aggregate_method([row(method=method, fraction=fraction, value=7.)], method)
        assert result['kind'] == 'single_budget' and result['macro']['primary'] == 7.
        assert 'auc' not in result['primary_metric']

def test_macro_ratio_and_paired_whole_year_indices():
    indices = shared_bootstrap(range(5))
    baseline, candidate = np.array([1., 100., 100., 100., 100.]), np.array([0., 95., 95., 95., 95.])
    c = compare_years(candidate, baseline, np.ones(5), np.ones(5), indices)
    assert c['relative_auc_reduction'] == pytest.approx(21/401)
    assert c['relative_auc_reduction'] != pytest.approx(np.mean((baseline-candidate)/baseline))
    paired = compare_years(baseline-1, baseline, np.ones(5), np.ones(5), indices)
    assert paired['descriptive_95pct']['auc_reduction'] == pytest.approx([1., 1.])
    assert indices.shape == (2000, 5) and np.array_equal(indices, shared_bootstrap(range(5)))

def test_small_denominator_is_not_a_percentage_pass():
    indices = shared_bootstrap(range(5))
    comparison = compare_years(np.zeros(5), np.full(5, 1e-10), np.ones(5), np.ones(5), indices)
    assert comparison['relative_auc_reduction'] is None
    assert comparison['descriptive_95pct']['relative_auc_reduction'] is None
    assert comparison['relative_interval_invalid_draws'] == 2000
    assert support_status(comparison, criteria())['status'] == 'NOT_DETERMINABLE'
    gain = {'macro_delta_G': 1., 'positive_years': 5}
    assert support_status(comparison, criteria(True), gain)['status'] == 'SUPPORTED_EXPLORATORY'

def test_gain_sign_and_retention_gate_do_not_follow_final_score_alone():
    indices = shared_bootstrap(range(5))
    contrast = compare_years(np.full(5, 8.), np.full(5, 9.), np.full(5, .8), np.full(5, .8), indices)
    gain = incremental_gain(np.full(5, 8.5), np.full(5, 8.), np.full(5, 10.), np.full(5, 9.), indices)
    assert gain['macro_delta_G'] == -.5
    assert support_status(contrast, criteria(True), gain)['status'] == 'NOT_SUPPORTED'
    contrast['macro_retention_difference'] = -.01
    assert support_status(contrast, criteria())['status'] == 'NOT_SUPPORTED'

def test_missing_rows_do_not_produce_candidate_comparisons_and_duplicates_fail():
    one, two = row(), row(fraction='0.2')
    expected = {identity(one): {}, identity(two): {}}
    result = summarize_records([one], expected, {})
    assert result['state'] == 'WAITING_EVIDENCE' and result['completeness']['missing_count'] == 1
    assert 'methods' not in result and 'claims' not in result
    with pytest.raises(ValueError, match='重复'):
        validate_completeness([one, one], expected)
    wrong = row(seed=3037)
    with pytest.raises(ValueError, match='计划之外'):
        validate_completeness([wrong], expected)

def synthetic_complete_table():
    values = {'C0': 100., 'C1': 100., 'C2': 100., 'C3': 100., 'N1': 90., 'N1_INT': 100.,
              'N1_ZERO': 120., 'C2_ZERO': 110., 'N1_ALL': 80., 'C2_ALL': 90.}
    records = []
    for method in (*CURVES, *POINTS):
        for year in range(2020, 2025):
            for role in [11, 23, 37, 53, 71]:
                for seed in ['NA'] if method == 'C0' else [1009, 2027, 3037]:
                    for fraction in [POINTS[method]] if method in POINTS else FRACTIONS:
                        records.append(row(method, year, role, seed, fraction, values[method]))
    config = {'data': {'outer_years': list(range(2020, 2025))},
              'metrics': {'bootstrap': {'draws': 2000, 'seed': 12602}, 'zero_denominator': 1e-8,
                          'aggregation': ['环境', '网络种子', '角色', '年份']},
              'support': {'V126_H1': {**criteria(), 'comparator': 'C2'},
                          'V126_H2': {**criteria(), 'comparator': 'C3'},
                          'V126_H3': {**criteria(True), 'comparator': 'N1_INT'}}}
    return records, {identity(r): {} for r in records}, config

def test_complete_1900_synthetic_identities_support_four_separate_claims():
    records, expected, config = synthetic_complete_table()
    assert len(records) == len(expected) == 1900
    result = summarize_records(records, expected, config)
    assert result['state'] == 'CORE_EVIDENCE_SUMMARIZED'
    assert set(result['claims']) == {'V126_H1', 'V126_H2', 'V126_H3', 'V126_PRACTICAL'}
    assert all(c['status'] == 'SUPPORTED_EXPLORATORY' for c in result['claims'].values())
    assert result['incremental_gain']['macro_delta_G'] == 20.
    assert result['methods']['N1']['macro']['primary'] == pytest.approx(90.)
    assert result['bootstrap']['same_indices_for_all_methods_and_comparisons']
    json.dumps(result, allow_nan=False)

def test_complete_table_rejects_mixed_audit_ids_and_budget():
    records, expected, config = synthetic_complete_table()
    records[0]['audit_ids_sha256'] = 'different_audit'
    with pytest.raises(ValueError, match='审计'):
        summarize_records(records, expected, config)
    records[0]['audit_ids_sha256'] = records[1]['audit_ids_sha256']
    records[0]['budget'] += 1
    with pytest.raises(ValueError, match='预算'):
        summarize_records(records, expected, config)

def test_same_budget_cannot_mix_different_support_names():
    records, expected, config = synthetic_complete_table()
    records[0]['support_ids_sha256'] = 'different_support'
    with pytest.raises(ValueError, match='名单'):
        summarize_records(records, expected, config)

def test_fractional_tail_differs_from_quantile():
    envs = [{'environment': str(i), 'count': 1, 'standardized_regret': float(i), 'retention': .5} for i in range(7)]
    result = environment_means(envs)
    assert result['environment_cvar_0.8'] == pytest.approx((6+.4*5)/1.4)
    assert result['environment_quantile_0.8'] != pytest.approx(result['environment_cvar_0.8'])

def test_c0_loader_rejects_old_2024_even_if_filename_and_identity_match(tmp_path):
    path, auditpath = tmp_path/'c0.json', tmp_path/'M_audit.json'
    saved = {'state': 'PASS', 'method': 'M', 'model_spec': 'full', 'year': 2024, 'seed': 11, 'fraction': '0.1',
             'budget': 10, 'query_count': 101, 'audit_ids': ['id'], 'environment_metrics': row()['environment_metrics']}
    path.write_text(json.dumps(saved), encoding='utf-8')
    audit = {'state': 'PASS', 'M_count': 120, 'years': [{'year': 2024, 'groups': [{'seed': 11,
             'M_sources': [{'fraction': '0.1', 'source': str(path), 'sha256': sha(path)}]}]}]}
    auditpath.write_text(json.dumps(audit), encoding='utf-8')
    sources = {'all120_M_identity_audit': {'path': str(auditpath), 'sha256': sha(auditpath)},
               'prepared': {'sha256': 'new'}, 'input_receipt': {'sha256': 'binding'}}
    expected = {('C0', 2024, 11, 'NA', '0.1'): {'source_path': str(path)}}
    with pytest.raises(ValueError, match='旧2024'):
        load_c0(expected, sources)
