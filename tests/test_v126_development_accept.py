"""开发完整性收尾的定向污染测试，不读取真实实验。"""
import copy
import json
import numpy as np
import pytest
import torch

from operations.accept_v126_development import (
    check_initial_inheritance, sha, validate_prediction_chain,
)


def evidence_pair(tmp_path):
    prediction = tmp_path / 'b0.1.npz'
    np.savez(prediction, combo_ids=np.array(['id']), prediction=np.array([1.25]))
    evidence = {'checkpoint_model_sha256': 'current_model', 'prediction_sha256': sha(prediction),
                'support_sha256': 'support'}
    prediction.with_suffix('.receipt.json').write_text(json.dumps(evidence), encoding='utf-8')
    summary = {'prediction_sha256': sha(prediction),
               'prediction_receipt_sha256': sha(prediction.with_suffix('.receipt.json')),
               'support_sha256': 'support'}
    return prediction, evidence, summary


def test_current_prediction_chain_passes_but_old_epoch_is_rejected(tmp_path):
    path, evidence, summary = evidence_pair(tmp_path)
    validate_prediction_chain(path, evidence, summary, 'current_model')
    with pytest.raises(ValueError, match='当前轮'):
        validate_prediction_chain(path, evidence, summary, 'another_epoch_model')


@pytest.mark.parametrize('key', ['prediction_sha256', 'prediction_receipt_sha256', 'support_sha256'])
def test_summary_must_bind_prediction_receipt_and_support(tmp_path, key):
    path, evidence, summary = evidence_pair(tmp_path)
    summary[key] = 'broken_link'
    with pytest.raises(ValueError):
        validate_prediction_chain(path, evidence, summary, 'current_model')


def test_initial_optimizer_only_allows_declared_learning_rate_change():
    source = {'model': {'weight': torch.tensor([1., 2.])},
              'optimizer': {'state': {0: {'exp_avg': torch.tensor([.1, .2])}},
                            'param_groups': [{'lr': .001, 'params': [0], 'weight_decay': .0001}]}}
    initial = copy.deepcopy(source)
    initial['optimizer']['param_groups'][0]['lr'] = .0003
    check_initial_inheritance(initial, source)
    initial['optimizer']['state'][0]['exp_avg'][0] += .01
    with pytest.raises(ValueError, match='优化器'):
        check_initial_inheritance(initial, source)


def test_initial_model_must_match_source_A():
    source = {'model': {'weight': torch.tensor([1.])}, 'optimizer': {'state': {}, 'param_groups': [{'lr': .001}]}}
    initial = copy.deepcopy(source)
    initial['optimizer']['param_groups'][0]['lr'] = .0003
    initial['model']['weight'][0] += 1.
    with pytest.raises(ValueError, match='模型'):
        check_initial_inheritance(initial, source)
