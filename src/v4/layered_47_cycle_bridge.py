"""Scoped continuation protocol; leaves historical pilot bridge unchanged."""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np


def instrument(source):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('cycle_bridge_anchor:' + old[:70])
        source = source.replace(old, new)

    source = source.replace('operation == "init_loaded"',
                            '(operation == "init_loaded" || operation == "init_cycle")')
    replace('    double restored_bucket_integral_effort = 0.0;', '''    double restored_bucket_integral_effort = 0.0;
    double restored_active = 0, restored_clock = 0, restored_last_time = 0;
    double restored_dt = 1.0/4800.0;
    bool restore_native = false;
    std::vector<double> saved_warmstart, saved_qacc, saved_ctrl;''')
    replace('      array("bucket_active_curl_effort", bucket_active_curl_effort.data(), 4);', '''      array("bucket_active_curl_effort", bucket_active_curl_effort.data(), 4);
      array("cycle_native_warmstart", data->qacc_warmstart, model->nv);
      array("cycle_native_qacc", data->qacc, model->nv);
      array("cycle_native_ctrl", data->ctrl, model->nu);
      std::cout << ",\\"cycle_controller_last_time\\":" << controller.last_time
                << ",\\"cycle_native_timestep\\":" << model->opt.timestep;''')
    replace('        std::string extra;\n        if (input >> extra) throw std::runtime_error("extra init fields");', '''        if (operation == "init_cycle") {
          if (model->na != 0) throw std::runtime_error("cycle actuator state unsupported");
          restored_active = number();
          restored_clock = number();
          restored_last_time = number();
          restored_dt = number();
          const double native_flag = number();
          if (restored_active < 0 || restored_active > BucketActiveCurl::cap ||
              restored_clock < 0 || restored_last_time > restored_clock ||
              restored_dt <= 0 || std::abs(restored_dt-1.0/4800.0)>1e-12 ||
              (native_flag != 0 && native_flag != 1))
            throw std::runtime_error("cycle native envelope");
          restore_native = native_flag == 1;
          if (restore_native) {
            saved_warmstart.resize(model->nv); saved_qacc.resize(model->nv);
            saved_ctrl.resize(model->nu);
            for (double& x : saved_warmstart) x = number();
            for (double& x : saved_qacc) x = number();
            for (double& x : saved_ctrl) x = number();
          }
          data->time = restored_clock;
          model->opt.timestep = restored_dt;
        }
        std::string extra;
        if (input >> extra) throw std::runtime_error("extra init fields");''')
    replace('          bucket_integral_effort[3] = restored_bucket_integral_effort;', '''          bucket_integral_effort[3] = restored_bucket_integral_effort;
          if (operation == "init_cycle") {
            active_curl.effort = restored_active;
            bucket_active_curl_effort[3] = restored_active;
            controller.last_time = restored_last_time;
          }''')
    replace('        ready = true;\n        std::copy(data->xpos', '''        if (operation == "init_cycle" && restore_native) {
          std::copy(saved_warmstart.begin(), saved_warmstart.end(), data->qacc_warmstart);
          std::copy(saved_qacc.begin(), saved_qacc.end(), data->qacc);
          std::copy(saved_ctrl.begin(), saved_ctrl.end(), data->ctrl);
        }
        ready = true;
        std::copy(data->xpos''')
    return source


def native_restore_tokens(state, rock_count):
    active = np.asarray(state['bucket_active_curl_effort'], dtype=float)
    if (active.shape != (4,) or not np.isfinite(active).all()
            or np.any(active[:3] != 0) or not 0 <= active[3] <= 60000):
        raise ValueError('cycle_active_effort')
    t, last, dt = [float(state[k]) for k in (
        'native_time_s', 'native_last_time_s', 'native_timestep_s')]
    if (not np.isfinite([t,last,dt]).all() or t < 0 or last > t
            or abs(dt-1/4800) > 1e-12 or t-last > dt+1e-8):
        raise ValueError('cycle_native_clock')
    full = state['native_state_kind'] == 'full'
    if state['native_state_kind'] not in ('full', 'legacy_diagnostic_cold_acceleration'):
        raise ValueError('cycle_native_state_kind')
    tokens = [active[3], t, last, dt, int(full)]
    if full:
        for key, size in [('native_warmstart',18+6*rock_count),
                          ('native_qacc',18+6*rock_count), ('native_ctrl',12)]:
            values=np.asarray(state[key],dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError('cycle_native_array:'+key)
            tokens.extend(values.tolist())
    return tokens


def controller_snapshot(state):
    return {
        **{k:list(state[k]) for k in ('controller_target',
           'controller_target_velocity', 'bucket_integral_effort',
           'bucket_active_curl_effort', 'bucket_linear_velocity_world_m_s',
           'bucket_angular_velocity_world_rad_s')},
        'native_state_kind':'full', 'native_time_s':float(state['time_s']),
        'native_last_time_s':float(state['cycle_controller_last_time']),
        'native_timestep_s':float(state['cycle_native_timestep']),
        **{k:list(state[v]) for k,v in (
            ('native_warmstart','cycle_native_warmstart'),
            ('native_qacc','cycle_native_qacc'), ('native_ctrl','cycle_native_ctrl'))},
    }


@contextmanager
def installed_cycle_bridge(build_dir):
    import layered_mound_checkpoint_runtime as runtime
    import run_layered_47_interactive_proxy_benchmark as benchmark
    original_benchmark_class = benchmark.RestoredRuntimeBridge
    original_render = runtime.render_capacity_bridge
    original_class = runtime.RestoredRuntimeBridge
    destination=Path(build_dir);destination.mkdir(parents=True,exist_ok=False)

    class CycleBridge(original_class):
        def __init__(self, binary, model, count, qpos, qvel, rock_qpos,
                     rock_qvel, controller_state=None):
            self.cycle_tokens = (None if controller_state is None else
                                 native_restore_tokens(controller_state,count))
            super().__init__(binary,model,count,qpos,qvel,rock_qpos,rock_qvel,
                             controller_state)

        def _write(self, line):
            if line.startswith('init_loaded ') and self.cycle_tokens is not None:
                line='init_cycle '+line[len('init_loaded '):].rstrip()+' '+ ' '.join(
                    f'{float(x):.17g}' for x in self.cycle_tokens)+'\n'
            return super()._write(line)

    def render(*args):
        source,manifest=original_render(*args)
        source=instrument(source)
        (destination/'capacity-bridge.cc').write_text(source)
        return source,dict(manifest,cycle_continuation_protocol={
            'schema':'cycle_native_restore_v1','source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'full_native_state_saved':True})
    runtime.render_capacity_bridge=render
    runtime.RestoredRuntimeBridge=CycleBridge
    benchmark.RestoredRuntimeBridge=CycleBridge
    try:yield
    finally:
        runtime.render_capacity_bridge=original_render
        runtime.RestoredRuntimeBridge=original_class
        benchmark.RestoredRuntimeBridge=original_benchmark_class
