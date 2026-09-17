"""Phase 10A corruption regressions. Every record in this module is synthetic."""
import ast
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from recall_aware_ids.experiment.caching import (
    AttackCacheBuilder, ConcreteAttackCacheProvider, calculate_file_hash,
)
from recall_aware_ids.experiment.schemas import CompletionMarker, _REQUIRED_PROVENANCE_KEYS
from recall_aware_ids.experiment.boundary_selection import select_boundary_targets
from recall_aware_ids.experiment.matrix import generate_evaluation_matrix

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def cache_case(tmp_path):
    directory = tmp_path / 'cache'
    directory.mkdir()
    names = [f'f{i}' for i in range(78)]
    x = pd.DataFrame(np.zeros((500, 78), dtype=np.float32), columns=names)
    status = pd.DataFrame({
        'eval_position': np.arange(500), 'eligible': False, 'attempted': False,
        'successful': False, 'status_code': 'NOT_APPLICABLE',
        'queries_used': np.zeros(500, dtype=np.int64),
        'l0': 0., 'l1': 0., 'l2': 0., 'linf': 0.,
    })
    mapping = pd.DataFrame({'eval_position': np.arange(500), 'batch_id': 0})
    manifest = {key: digest(key) for key in _REQUIRED_PROVENANCE_KEYS}
    manifest.update({key: digest(key) for key in (
        'X_eval_hash', 'metadata_eval_hash', 'crafting_identity_hash',
        'measurement_identity_hash',
    )})
    manifest.update(attack_scenario='SilentProbing', effective_seed=42,
                    schema_version='1.0', row_count=500,
                    attack_script_hashes={'generator.py': digest('generator')},
                    attack_parameters={'modifies_samples': False},
                    query_budgets={'max_queries_per_sample': 0})

    def write():
        x.to_parquet(directory / 'X_attacked.parquet', index=False)
        status.to_parquet(directory / 'status.parquet', index=False)
        manifest['X_attacked_sha256'] = calculate_file_hash(directory / 'X_attacked.parquet')
        manifest['output_sha256'] = manifest['X_attacked_sha256']
        manifest['status_sha256'] = calculate_file_hash(directory / 'status.parquet')
        (directory / 'manifest.json').write_text(json.dumps(manifest))

    def load(**extra):
        return ConcreteAttackCacheProvider(directory, mapping, manifest, names,
                                           official_mode=False, expected_row_count=500, **extra)
    write()
    return x, status, mapping, manifest, write, load


@pytest.mark.parametrize('bad', ['', 'unknown', 'a'*64, '0'*64, 'not-a-hash', None])
def test_v5_provider_rejects_malformed_identity(cache_case, bad):
    x, status, mapping, manifest, write, load = cache_case
    manifest['scaler_hash'] = bad
    write()
    with pytest.raises((ValueError, TypeError)):
        load()


@pytest.mark.parametrize('field', ['attack_script_hashes', 'attack_parameters', 'query_budgets'])
def test_v5_provider_rejects_empty_nested_identity(cache_case, field):
    *_, manifest, write, load = cache_case
    manifest[field] = {}
    write()
    with pytest.raises((ValueError, TypeError)):
        load()


@pytest.mark.parametrize('bad', [True, 0.5, float('nan'), '1'])
def test_v5_provider_rejects_query_types(cache_case, bad):
    x, status, mapping, manifest, write, load = cache_case
    status['queries_used'] = bad
    write()
    with pytest.raises((ValueError, TypeError)):
        load()


def test_v5_provider_rejects_scrambled_identity(cache_case):
    x, status, mapping, manifest, write, load = cache_case
    status['eval_position'] = np.arange(499, -1, -1)
    write()
    with pytest.raises(ValueError):
        load()


def test_v5_provider_rejects_string_features(cache_case):
    x, status, mapping, manifest, write, load = cache_case
    x['f0'] = '0.0'
    write()
    with pytest.raises((ValueError, TypeError)):
        load()


def test_v5_provider_rejects_silent_success(cache_case):
    x, status, mapping, manifest, write, load = cache_case
    status.loc[0, ['eligible', 'attempted', 'successful']] = True
    status.loc[0, 'status_code'] = 'SUCCESS'
    write()
    with pytest.raises(ValueError):
        load()


def test_v5_provider_cannot_disable_boolean_validation(cache_case):
    x, status, mapping, manifest, write, load = cache_case
    status['eligible'] = 0
    write()
    with pytest.raises((ValueError, TypeError)):
        load(validate_strict_bool=False)


def test_v5_builder_rejects_incomplete_provenance():
    with pytest.raises((ValueError, TypeError)):
        AttackCacheBuilder([f'f{i}' for i in range(78)], [True]*63+[False]*15,
                           pd.DataFrame({'train_min': [-1.]*78, 'train_max': [1.]*78}),
                           {'frozen_rf_hash': digest('model')}, 42)


def test_v5_boundary_official_target_count():
    with pytest.raises(ValueError):
        select_boundary_targets(np.ones(72000, dtype=bool), 42, 199, official_mode=True)


@pytest.mark.parametrize('timestamp', ['', 'yesterday', '2026-99-99T25:00:00Z'])
def test_v5_completion_timestamp(timestamp):
    with pytest.raises((ValueError, TypeError)):
        CompletionMarker('run', timestamp, {k: digest(k) for k in _REQUIRED_PROVENANCE_KEYS})


def test_v5_matrix_rejects_c8_replacing_c7(tmp_path):
    for file in (ROOT / 'configs').glob('*.yaml'):
        (tmp_path / file.name).write_bytes(file.read_bytes())
    p = tmp_path / 'controllers.yaml'
    obj = yaml.safe_load(p.read_text())
    obj['controller_configurations']['C8'] = obj['controller_configurations'].pop('C7')
    p.write_text(yaml.safe_dump(obj))
    with pytest.raises(ValueError):
        generate_evaluation_matrix(tmp_path)


@pytest.mark.parametrize('relative', ['src/recall_aware_ids/experiment/caching.py', 'scripts/run_m2_pilot.py'])
def test_v5_no_untracked_target_calls(relative):
    tree = ast.parse((ROOT / relative).read_text())
    violations = [node.lineno for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id in {'predict_fn', 'predict_wrapper'}]
    assert not violations, f'Untracked target calls at {relative}:{violations}'
