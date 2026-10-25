from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import copy
import json
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/v4'))
import run_layered_sparse_a30_probe as probe
import layered_frequency_bridge as bridge
import run_layered_sparse_resume_audit as resume


class SparseA30ProbeTest(unittest.TestCase):
    def test_both_frequencies_measure_equal_sim_time(self):
        for hz in (60, 120):
            p = probe.policy(hz)
            self.assertEqual(p['warmup_max']/hz, 2)
            self.assertEqual(p['required_strict_tail']/hz, .5)
            self.assertEqual(p['measured_frames']/hz, 1)

    def test_grid_then_frequency_in_remote_wrapper(self):
        root = Path(__file__).resolve().parents[2]
        text = (root/'scripts/a30_sparse_probe/run_probe.sh').read_text()
        self.assertIn('for hz in 120 60;', text)
        self.assertIn('--solver sparse10', text)
        self.assertIn('--seed-candidate-report "$seed_report" --seed-resume-report "$seed_audit"', text)
        self.assertIn('LAYERED_SPARSE_120_REPORT', text)

    def test_subcycle_preserves_impulse_and_first_half_diagnostics(self):
        calls = []
        def step(b, dt, substeps, command, wrench):
            calls.append((dt, substeps, wrench))
            return dict(contact_substeps={'rock_bucket': int(len(calls) == 1)},
                        forbidden_contact_count=int(len(calls) == 1), closure_m=.001/len(calls))
        forces = {'rock': {'force': [12., 0., 0.]}}
        with patch.object(bridge, '_bridge_step', side_effect=step):
            result = bridge.subcycled_step(None, 1/60, 80, np.zeros(4), forces)
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(sum(c[0]*c[2]['rock']['force'][0] for c in calls), 12/60)
        self.assertEqual(result['contact_substeps']['rock_bucket'], 1)
        self.assertEqual(result['forbidden_contact_count'], 1)
        self.assertEqual(result['closure_m'], .001)

    def test_rejects_unapproved_frequency(self):
        with self.assertRaises(ValueError):
            probe.policy(30)
        with self.assertRaises(ValueError):
            bridge.subcycled_step(None, 1/60, 40, np.zeros(4), {})
    def test_separate_47_65_routes(self):
        for count in (47, 65):
            loader, selected = probe.select(count)
            self.assertEqual(len(selected['expected_ids']), count)
            self.assertIn(str(count), loader.__module__)
        with self.assertRaises(ValueError):
            probe.select(33)

    def test_solver_change_is_explicit_and_keeps_iterations(self):
        base = {'mpm': {'voxel_size_m': .05}}
        fixed = probe.profile(base, 'fixed5')
        sparse = probe.profile(base, 'sparse10')
        self.assertEqual(sparse['voxel_size_m'], .1)
        self.assertEqual(sparse['grid_type'], 'sparse')
        self.assertEqual(sparse['grid_padding'], 2)
        for key in fixed.keys() - {'voxel_size_m', 'grid_type', 'grid_padding'}:
            self.assertEqual(fixed[key], sparse[key])

    def test_nonfinite_contact_or_displacement_stops_diagnostic(self):
        row = dict(finite=True, forbidden_contact_count=0, external_rigid_support_substeps=0,
                   unmapped_active_collider_sample_count=0, nonfinite_collider_sample_count=0,
                   bucket_mpm_contact_samples=0, rock_bucket_contact_substeps=0, closure_m=.0001,
                   maximum_rock_linear_speed_m_s=.02, maximum_rock_angular_speed_rad_s=.1,
                   particle_max_speed_m_s=.03, particle_p99_speed_m_s=.02, rock_displacement_from_source_m=.001)
        self.assertTrue(probe.guard(row))
        for key, value in [('finite', False), ('bucket_mpm_contact_samples', 1), ('rock_displacement_from_source_m', .006)]:
            self.assertFalse(probe.guard({**row, key: value}))

    def test_historical_grid_transient_is_not_strict_or_measurable(self):
        row = dict(finite=True, stable=False, forbidden_contact_count=0,
                   external_rigid_support_substeps=0, unmapped_active_collider_sample_count=0,
                   nonfinite_collider_sample_count=0, bucket_mpm_contact_samples=0,
                   rock_bucket_contact_substeps=0, closure_m=.0001,
                   maximum_rock_linear_speed_m_s=.027, maximum_rock_angular_speed_rad_s=.176,
                   particle_max_speed_m_s=.072, particle_p99_speed_m_s=.0585627034,
                   rock_displacement_from_source_m=.001)
        self.assertTrue(probe.guard(row, 'warmup'))
        self.assertFalse(probe.guard(row, 'measured'))
        self.assertEqual(probe.classify_frame(row, 'warmup', 20), 0)
        self.assertFalse(row['strict'])
        frequency_row = row.copy()
        self.assertEqual(probe.classify_frame(frequency_row, 'warmup', 20, 'frequency'), 0)
        self.assertEqual(frequency_row['guard_failures'], ['particle_p99_speed_m_s'])
        for key, value in [('finite', False), ('bucket_mpm_contact_samples', 1),
                           ('rock_displacement_from_source_m', .006), ('particle_max_speed_m_s', .51),
                           ('maximum_rock_linear_speed_m_s', .11), ('closure_m', float('nan'))]:
            unsafe = {**row, key: value}
            self.assertEqual(probe.classify_frame(unsafe, 'warmup', 20), 0)
            self.assertIn(key, unsafe['guard_failures'])

    def test_strict_tail_only_accumulates_stable_frames(self):
        with patch.object(probe, 'guard_checks', return_value={'finite': True}):
            row = {'stable': True}
            self.assertEqual(probe.classify_frame(row, 'warmup', 4), 5)
            self.assertTrue(row['strict'])
            row['stable'] = False
            self.assertEqual(probe.classify_frame(row, 'measured', 5), 0)
            self.assertFalse(row['strict'])

    def test_sparse60_cannot_skip_audited_seed(self):
        for hz, solver, seed, audit in [(60, 'sparse10', None, None),
                (60, 'sparse10', 'seed', None), (120, 'sparse10', 'seed', 'audit'),
                (60, 'fixed5', 'seed', 'audit')]:
            with self.assertRaises(ValueError):
                probe.validate_seed_request(SimpleNamespace(hz=hz, solver=solver,
                    seed_candidate_report=seed, seed_resume_report=audit))
        probe.validate_seed_request(SimpleNamespace(hz=60, solver='sparse10',
            seed_candidate_report='seed', seed_resume_report='audit'))
        probe.validate_seed_request(SimpleNamespace(hz=120, solver='sparse10'))

    def test_independent_audit_must_bind_same_seed_and_all_checks(self):
        path = Path('/test/120hz.json')
        before = {'path': '/test/checkpoint.npz', 'sha256': 'checkpoint'}
        source = {'policy': {'hz': 120}}
        audit = dict(schema='layered_sparse_resume_audit_v1', status='sparse_resume_audit_passed',
            rocks=47, hz=120, checks={k: True for k in resume.AUDIT_CHECKS},
            runner_sha256='hash', candidate_report=str(path), candidate_report_sha256='hash',
            candidate_checkpoint=before, execution={'frames': 240}, trace=[{'strict': True}] * 240)
        with patch.object(probe.bench, 'sha256', return_value='hash'):
            resume.validate_seed_audit(audit, path, source, before, 47, 120)
            for field, value in [('checks', {}), ('candidate_report_sha256', 'wrong'),
                    ('candidate_checkpoint', {}), ('rocks', 65), ('hz', 60),
                    ('runner_sha256', 'old'), ('trace', [{'strict': True}]),
                    ('status', 'sparse_resume_audit_rejected')]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    resume.validate_seed_audit({**audit, field: value}, path, source, before, 47, 120)
            changed = copy.deepcopy(audit)
            changed['trace'][0]['strict'] = False
            with self.assertRaises(ValueError):
                resume.validate_seed_audit(changed, path, source, before, 47, 120)

    def test_invalid_seed_writes_failure_report_before_loading_source(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / 'failure.json'
            args = SimpleNamespace(output=str(output), hz=60, rocks=47, solver='sparse10',
                device='cuda:0', seed_candidate_report='incomplete', seed_resume_report=None)
            with patch.object(probe.bench, 'load_checkpoint_readonly') as loader:
                result = probe.run(args)
            loader.assert_not_called()
            self.assertEqual(result['status'], 'probe_failed')
            self.assertEqual(result['execution']['warmup_frames'], 0)
            self.assertEqual(json.loads(output.read_text())['failure']['message'],
                             'seed_requires_candidate_and_independent_audit')

    def test_seed_report_checkpoint_or_metadata_change_is_rejected(self):
        identity = {'path': '/test/seed.npz', 'sha256': 'checkpoint'}
        seed = dict(checkpoint=identity, metadata_sha256='hash',
                    candidate_report='/test/seed.json', candidate_report_sha256='hash',
                    resume_report='/test/audit.json', resume_report_sha256='hash')
        with patch.object(probe.bench, 'checkpoint_identity', return_value=identity), \
                patch.object(probe.bench, 'sha256', return_value='hash'):
            self.assertTrue(probe.seed_unchanged(seed))
            for field in ('metadata_sha256', 'candidate_report_sha256', 'resume_report_sha256'):
                self.assertFalse(probe.seed_unchanged({**seed, field: 'changed'}))
        with patch.object(probe.bench, 'checkpoint_identity', return_value={**identity, 'sha256': 'changed'}):
            self.assertFalse(probe.seed_unchanged(seed))
