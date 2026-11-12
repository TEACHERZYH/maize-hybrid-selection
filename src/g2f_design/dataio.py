"""按已验收行号读取原始数据；最小样本入口可限制组合与读取范围。"""
import calendar
import csv
import datetime
import json
from pathlib import Path
import unicodedata
import numpy as np


def identifier(value):
    return unicodedata.normalize('NFC', value.strip())


def numeric(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (ValueError, TypeError):
        return np.nan


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        yield from csv.DictReader(handle)


def ledger_rows(contract, maximum_year=None):
    for row in csv_rows(Path(contract) / 'combination_cost_ledger.csv'):
        if maximum_year is None or int(row['year']) <= maximum_year:
            row['year'] = int(row['year'])
            row['source_row_indices'] = json.loads(row['source_row_indices'])
            yield row


def read_plots(raw, combinations):
    required = {}
    for combo in combinations:
        for index in combo['source_row_indices']:
            if index in required:
                raise ValueError('同一真实小区属于多个组合')
            required[index] = (combo['env'], combo['hybrid'])
    plots = {}
    for index, row in enumerate(csv_rows(Path(raw) / '1_Training_Trait_Data_2014_2023.csv')):
        if index not in required:
            continue
        key = (identifier(row['Env']), identifier(row['Hybrid']))
        if key != required[index]:
            raise ValueError('原始数据与冻结行号不一致')
        value = numeric(row['Yield_Mg_ha'])
        if not np.isfinite(value):
            raise ValueError('冻结有效小区出现缺失产量')
        block = [identifier(row[k]) for k in ('Env', 'Experiment', 'Replicate', 'Block')]
        plots[index] = {'value': value, 'block': json.dumps(block, ensure_ascii=False, separators=(',', ':')),
                        'plot_id': json.dumps(block + [identifier(row['Plot'])], ensure_ascii=False, separators=(',', ':'))}
        if len(plots) == len(required):
            break
    if len(plots) != len(required):
        raise ValueError('冻结小区读取不完整')
    return plots


def combination_values(combinations, plots):
    return np.array([np.mean([plots[i]['value'] for i in row['source_row_indices']]) for row in combinations])


def read_genotypes(raw, hybrids):
    required, found = set(hybrids), {}
    path = Path(raw) / '5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt'
    with path.open(encoding='utf-8-sig') as handle:
        if handle.readline().strip() != '<Numeric>':
            raise ValueError('不是冻结Numeric格式')
        markers = handle.readline().strip().split('\t')[1:]
        for line in handle:
            cells = line.rstrip('\r\n').split('\t')
            key = identifier(cells[0])
            if key not in required:
                continue
            if key in found or len(cells) != len(markers) + 1:
                raise ValueError('基因型重复或位点数量变化')
            if any(c not in ('0', '0.5', '1', 'NA') for c in cells[1:]):
                raise ValueError('基因型编码改变')
            found[key] = np.array([numeric(c) * 2 for c in cells[1:]])
            if len(found) == len(required):
                break
    if set(found) != required:
        raise ValueError('缺少冻结基因型')
    return found, markers


def read_environment_features(raw, contract, environments):
    raw, contract = Path(raw), Path(contract)
    rules = json.loads((contract / 'feature_whitelist.json').read_text(encoding='utf-8'))
    availability = {r['env']: r for r in json.loads((contract / 'environment_availability.json').read_text(encoding='utf-8'))}
    needed = set(environments)
    sources = set(e for env in needed for e in availability[env]['historical_climate_source_envs'])
    meta = {identifier(r['Env']): r for r in csv_rows(raw / '2_Training_Meta_Data_2014_2023.csv') if identifier(r['Env']) in needed}
    soil = {identifier(r['Env']): r for r in csv_rows(raw / '3_Training_Soil_Data_2015_2023.csv') if identifier(r['Env']) in needed}
    weather = {}
    for row in csv_rows(raw / '4_Training_Weather_Data_2014_2023_full_year.csv'):
        env = identifier(row['Env'])
        if env not in sources:
            continue
        date = datetime.datetime.strptime(row['Date'], '%Y%m%d').date()
        if date.year != availability[env]['year']:
            raise ValueError('天气来源年份与已验收映射不符')
        weather.setdefault((env, date.month), []).append([numeric(row[c]) for c in ('T2M', 'PRECTOTCORR')])
    monthly = {}
    for (env, month), values in weather.items():
        array = np.asarray(values)
        days = calendar.monthrange(availability[env]['year'], month)[1]
        result = []
        for col in range(2):
            valid = array[np.isfinite(array[:, col]), col]
            if col == 1 and np.any(valid < 0):
                raise ValueError('降水出现负值，需核实源数据缺失码')
            result.append((np.mean(valid) if col == 0 else np.sum(valid)) if len(valid) * 10 >= days * 9 else np.nan)
        monthly[(env, month)] = result
    features = {}
    for env in sorted(needed):
        row, rule = meta[env], availability[env]
        prior = rule['historical_climate_source_envs']
        if any(availability[e]['year'] >= rule['year'] for e in prior):
            raise ValueError('环境特征引用当年或未来天气')
        values = [numeric(row[c]) for c in rules['geographic_columns']]
        values += [numeric(soil.get(env, {}).get(c)) if rule['soil_available_before_first_planting'] else np.nan
                   for c in rules['soil_columns']]
        for variable in range(2):
            for month in range(1, 13):
                observed = [monthly.get((e, month), [np.nan, np.nan])[variable] for e in prior]
                observed = [v for v in observed if np.isfinite(v)]
                values.append(np.mean(observed) if observed else np.nan)
        features[env] = np.asarray(values, dtype=np.float64)
    names = rules['geographic_columns'] + rules['soil_columns'] + [f'{v}_past3_month{m:02d}' for v in ('T2M', 'PRECTOTCORR') for m in range(1, 13)]
    return features, names
