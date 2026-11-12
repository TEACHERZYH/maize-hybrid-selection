"""验收已完成的2019开发分支；运行中只返回等待，不改变停止点。"""
import argparse
import ast
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/v126/R1_2019_s1009'
SNAPSHOT = ROOT / 'runs/v126/code_R1_2019'
STAGES = ('ordinary', 'centered', 'N1')
ROLES = (11, 23, 37, 53, 71)
FRACTIONS = ('0.1', '0.2', '0.4', '0.6')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot_constant(name):
    tree = ast.parse((SNAPSHOT / 'analysis_v126/train.py').read_text(encoding='utf-8'))
    values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    require(len(values) == 1 and isinstance(values[0], int), '快照训练常量不明确：' + name)
    return values[0]


def model_digest(state):
    """与已运行预测器的摘要定义相同，不实例化或执行网络。"""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        require(torch.is_tensor(value) and torch.isfinite(value).all().item(), '模型参数非法')
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def same_nested(left, right, label):
    if torch.is_tensor(left):
        require(torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left, right), label)
    elif isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(), label)
        for key in left:
            same_nested(left[key], right[key], label + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        require(type(left) is type(right) and len(left) == len(right), label)
        for a, b in zip(left, right):
            same_nested(a, b, label)
    else:
        require(type(left) is type(right) and left == right, label)


def check_initial_inheritance(payload, pretrain):
    same_nested(pretrain['model'], payload['model'], '初始模型未精确继承A')
    expected = copy.deepcopy(pretrain['optimizer'])
    for group in expected['param_groups']:
        group['lr'] = 3e-4
    same_nested(expected, payload['optimizer'], '初始优化器除B学习率外偏离A')


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return math.isfinite(value) if isinstance(value, (float, int)) else True


def validate_prediction_chain(prediction, evidence, budget_summary, model_hash):
    """拒绝串用旧轮预测及评价摘要指向另一份文件。"""
    receipt_path = prediction.with_suffix('.receipt.json')
    require(evidence['checkpoint_model_sha256'] == model_hash, '预测并非当前轮检查点模型')
    require(evidence['prediction_sha256'] == sha(prediction) == budget_summary['prediction_sha256'],
            '评价摘要与实际预测文件断链')
    require(budget_summary['prediction_receipt_sha256'] == sha(receipt_path), '评价摘要与预测回执断链')
    require(budget_summary['support_sha256'] == evidence['support_sha256'], '评价摘要支持源不一致')


def replay_stopping(scores):
    """独立重放含初始点、1e-8容差、至少5轮、patience5的规则。"""
    best, best_epoch, misses = None, None, 0
    states = []
    for epoch, score in enumerate(scores):
        require(math.isfinite(score), '开发评分非有限')
        require(not states or not states[-1]['stop'], '已满足早停后仍出现额外完整训练轮次')
        if best is None or score < best - 1e-8:
            best, best_epoch, misses = score, epoch, 0
        else:
            misses += 1
        states.append(dict(best_score=best, best_epoch=best_epoch, no_improvement=misses,
                           stop=epoch >= 5 and misses >= 5, last_score=score))
    return states


def accept(stage):
    folder = BASE / stage
    result_path = folder / 'stage_result.json'
    if not result_path.exists():
        return {'stage': stage, 'state': 'RUNNING_OR_AWAITING_RESULT'}
    result = read(result_path)
    require(result['stage'] == stage and result['outer_year'] == 2019 and result['seed'] == 1009,
            '开发身份不一致')
    if result['state'] != 'COMPLETED':
        return {'stage': stage, 'state': result['state']}
    completed = result['completed_epoch']
    require(5 <= completed <= 30, '完成轮次越界')
    receipt = read(folder / 'input_receipt.json')
    require(receipt['stage'] == stage and receipt['outer_year'] == 2019 and receipt['seed'] == 1009,
            '输入回执开发身份不一致')
    data_dir = Path(receipt['data_directory'])
    for name, expected in receipt['scientific_files_sha256'].items():
        if name == 'projection.npy':
            path = SNAPSHOT / 'design_v126/projection.npy'
        elif name in ('history.npz', 'target_features.npz') or name.startswith('roles/'):
            path = data_dir / name
        else:
            path = SNAPSHOT / name
        require(sha(path) == expected, '科学输入发生变化：' + name)
    with (folder / 'epoch_metrics.jsonl').open(encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    training = [r for r in records if 'training_loss' in r]
    evaluation = [r for r in records if 'development' in r]
    require([r['epoch'] for r in training] == list(range(1, completed + 1)), '训练轮次不连续')
    require([r['epoch'] for r in evaluation] == list(range(completed + 1)), '开发轮次不连续')
    require(all(math.isfinite(r[k]) for r in training for k in
                ('training_loss', 'gradient_norm_mean', 'gradient_norm_max', 'step_seconds_mean')),
            '训练日志包含非有限值')
    steps_per_epoch, pretrain_epochs = snapshot_constant('STEPS_PER_EPOCH'), snapshot_constant('PRETRAIN_EPOCHS')
    with (folder / 'step_metrics.jsonl').open(encoding='utf-8') as handle:
        steps = [json.loads(line) for line in handle if line.strip()]
    require([row['step'] for row in steps] == list(range(10, completed * steps_per_epoch + 1, 10)),
            '诊断步日志不连续')
    require(all(row['epoch'] == (row['step'] - 1) // steps_per_epoch + 1 and finite_tree(row)
                and row['learning_rate'] == 3e-4 for row in steps), '诊断日志身份、学习率或数值非法')
    states = replay_stopping([r['development']['score'] for r in evaluation])
    require(states[-1]['stop'] or completed == 30, '尚未达到科学停止条件')
    require(result['stopping'] == states[-1] and result['best_epoch'] == states[-1]['best_epoch'],
            '完成回执停止点不符合预定规则')
    checkpoint_hashes = {}
    model_hashes, evaluation_hashes = {}, {}
    a_result = read(BASE / 'pretrain/stage_result.json')
    require(a_result['state'] == 'COMPLETED' and a_result['stage'] == 'pretrain'
            and a_result['outer_year'] == 2019 and a_result['seed'] == 1009
            and a_result['completed_epoch'] == pretrain_epochs, 'A预训练尚未有效完成')
    a_path = BASE / 'pretrain' / f'checkpoint_{pretrain_epochs:03d}.pt'
    pretrain = torch.load(a_path, map_location='cpu', weights_only=False)
    require(pretrain['stage'] == 'pretrain' and pretrain['epoch'] == pretrain_epochs
            and pretrain['outer_year'] == 2019 and pretrain['seed'] == 1009, 'A检查点身份不一致')
    require(all(receipt['scientific_files_sha256'].get(key) == value
                for key, value in pretrain['scientific_files_sha256'].items()), 'A与B共享科学输入不一致')
    for epoch, row in enumerate(evaluation):
        checkpoint = folder / f'checkpoint_{epoch:03d}.pt'
        digest = sha(checkpoint)
        checkpoint_hashes[str(epoch)] = digest
        state = read(checkpoint.with_suffix('.state.json'))
        require(state['checkpoint_sha256'] == digest and state['stopping'] == states[epoch],
                '检查点与停止状态不匹配')
        require(row['stopping'] == states[epoch], '日志停止状态不匹配')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(payload['format'] == 'v1.26' and payload['stage'] == stage and payload['epoch'] == epoch
                and payload['seed'] == 1009 and payload['outer_year'] == 2019
                and payload['steps_per_epoch'] == steps_per_epoch, '检查点内部身份或步数不匹配')
        require(payload['scientific_files_sha256'] == receipt['scientific_files_sha256']
                and payload['projection_sha256'] == receipt['projection_sha256'], '检查点科学输入不一致')
        if epoch == 0:
            check_initial_inheritance(payload, pretrain)
        model_hash = model_digest(payload['model'])
        model_hashes[str(epoch)] = model_hash
        del payload
        directory = folder / 'development' / f'epoch_{epoch:03d}'
        summary = read(directory / 'evaluation.json')
        evaluation_hashes[str(epoch)] = sha(directory / 'evaluation.json')
        require(summary == row['development'], '独立评价文件与训练端回传不一致')
        require(summary['state'] == 'EXECUTION_VALID' and summary['year'] == 2019
                and summary['role_seeds'] == list(ROLES) and len(summary['roles']) == 5
                and summary['target_file_sha256'] == receipt['scientific_files_sha256']['target_features.npz'],
                '开发评价身份或完整性异常')
        roles = {item['seed']: item for item in summary['roles']}
        require(set(roles) == set(ROLES) and len(roles) == len(summary['roles']), '评价角色重复或缺失')
        require(abs(summary['score'] - sum(item['score'] for item in roles.values()) / len(ROLES)) < 1e-12,
                '五角色评分汇总不一致')
        for seed in ROLES:
            role = read(data_dir / f'roles/s{seed}/role.json')
            role_summary = roles[seed]
            audit_path = data_dir / f'evaluation/s{seed}/audit_labels.npz'
            require(role_summary['year'] == 2019 and role_summary['role_sha256'] == sha(data_dir / f'roles/s{seed}/role.json')
                    and role_summary['audit_file_sha256'] == sha(audit_path), '评价角色或隐藏审计源不一致')
            audit_ids_hash = hashlib.sha256(json.dumps(role['audit_ids'], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            require(role_summary['audit_ids_sha256'] == audit_ids_hash, '评价审计ID不一致')
            budget_summaries = {item['fraction']: item for item in role_summary['budgets']}
            require(set(budget_summaries) == set(FRACTIONS) and len(role_summary['budgets']) == 4,
                    '评价预算缺失或重复')
            xs, ys = [], []
            for fraction in FRACTIONS:
                prediction = directory / f's{seed}' / f'b{fraction}.npz'
                evidence = read(prediction.with_suffix('.receipt.json'))
                budget_summary = budget_summaries[fraction]
                validate_prediction_chain(prediction, evidence, budget_summary, model_hash)
                require(evidence['prediction_sha256'] == sha(prediction), '预测文件散列异常')
                require(evidence['year'] == 2019 and evidence['seed'] == seed
                        and evidence['fraction'] == fraction, '预测身份不一致')
                for kind, filename in [('role', 'role.json'), ('support', f'support_b{fraction}.json')]:
                    require(evidence[kind + '_sha256'] == receipt['scientific_files_sha256'][f'roles/s{seed}/{filename}'],
                            '预测角色或支持源不一致')
                support = read(data_dir / f'roles/s{seed}/support_b{fraction}.json')
                require(evidence['label_unit'] == 'historical_standardized' and evidence['budget'] == support['budget']
                        == role['budgets'][fraction] == budget_summary['budget'], '评价单位或实际预算不一致')
                require(budget_summary['query_count'] == len(role['query_ids'])
                        and budget_summary['actual_fraction'] == support['budget'] / len(role['query_ids'])
                        and budget_summary['support_ids_sha256'] == support['ids_sha256'], '评价预算横轴或支持ID不一致')
                with np.load(prediction, allow_pickle=False) as saved:
                    require(set(saved.files) == {'combo_ids', 'prediction'} and saved['prediction'].shape == (len(role['audit_ids']),)
                            and np.isfinite(saved['prediction']).all() and len(set(saved['combo_ids'].tolist())) == len(role['audit_ids'])
                            and set(saved['combo_ids'].tolist()) == set(role['audit_ids']), '预测ID或数值非法')
                xs.append(budget_summary['actual_fraction']); ys.append(budget_summary['macro_standardized_regret'])
            require(all(math.isfinite(value) for value in ys) and all(b > a for a, b in zip(xs, xs[1:])), '角色评分或实际横轴非法')
            auc = sum((a + b) * (right - left) / 2 for a, b, left, right in zip(ys, ys[1:], xs, xs[1:])) / (xs[-1] - xs[0])
            require(abs(role_summary['score'] - auc) < 1e-12, '角色实际预算AUC不一致')
    final = dict(state='PASS_DEVELOPMENT_EXECUTION', stage=stage, completed_epoch=completed,
                 best_epoch=result['best_epoch'], best_score=states[-1]['best_score'],
                 initial_score=evaluation[0]['development']['score'],
                 checkpoint_count=len(checkpoint_hashes), checkpoint_sha256=checkpoint_hashes,
                 result_sha256=sha(result_path), input_receipt_sha256=sha(folder / 'input_receipt.json'),
                 integrity_version=2, checkpoint_model_sha256=model_hashes, evaluation_sha256=evaluation_hashes,
                 step_diagnostics_count=len(steps), steps_per_epoch=steps_per_epoch,
                 A_checkpoint_sha256=sha(a_path), initial_model_and_optimizer_inheritance=True,
                 epoch_metrics_sha256=sha(folder / 'epoch_metrics.jsonl'), step_metrics_sha256=sha(folder / 'step_metrics.jsonl'),
                 operations_files_sha256={str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
                 checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 scientific_claim='仅开发执行与停止点有效，外层正向效果尚未评价',
                 downstream_state='WAITING_AUTHORIZATION' if stage == 'N1' and result['best_epoch'] == 0 else 'BRANCH_READY')
    legacy_path = folder / 'acceptance.json'
    if legacy_path.exists():
        old = read(legacy_path)
        require(old['result_sha256'] == final['result_sha256'] and old['checkpoint_sha256'] == checkpoint_hashes,
                '已验收产物发生变化，须单独调查')
        final['legacy_acceptance_sha256'] = sha(legacy_path)
    destination = folder / 'acceptance_integrity_v2.json'
    if destination.exists():
        old = read(destination)
        require(all(old[key] == final[key] for key in final if key not in ('checked_utc', 'operations_files_sha256')),
                '完整性验收产物发生变化，须单独调查')
        return old
    destination.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES)
    args = parser.parse_args()
    for stage in (args.stage,) if args.stage else STAGES:
        result = accept(stage)
        print(json.dumps({k: result[k] for k in ('stage', 'state', 'completed_epoch', 'best_epoch',
                          'best_score', 'downstream_state') if k in result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
