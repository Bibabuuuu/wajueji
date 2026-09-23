"""Opt-in experiment: prune exact-zero tangent terms, never change physics.

Generate a new header from the hash-bound historical controller; do not edit
the evidence header, v3 producers, solver settings, or default viewer.
"""
from hashlib import sha256
from pathlib import Path

BASELINE_SHA256 = "6c0f9e33faa839c38e1c026d63e052d3f6aed8bd517899a3997279ddad52fb11"
MASS_OLD = """    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int u=0;u<m->nv;++u)for(int w=0;w<m->nv;++w)
      out.mass[i*4+j]+=T[u*4+i]*M[u*m->nv+w]*T[w*4+j];"""
MASS_NEW = """    // Exact structural zeros only: no tolerance, dropped DOFs, or cached forces.
    // Keep ascending indices and the original multiplication/addition order.
    // Fail closed before pruning, so zero*NaN cannot silently disappear.
    for(const auto value:M)if(!std::isfinite(value))throw std::runtime_error("inertia: nonfinite full mass");
    std::array<std::vector<int>,4> support;
    for(int u=0;u<m->nv;++u)for(int i=0;i<4;++i){
      if(!std::isfinite(T[u*4+i]))throw std::runtime_error("inertia: nonfinite tangent");
      if(T[u*4+i]!=0) support[i].push_back(u);
    }
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int u:support[i])for(int w:support[j])
      out.mass[i*4+j]+=T[u*4+i]*M[u*m->nv+w]*T[w*4+j];"""
EFFORT_OLD = """    for(int i=0;i<4;++i)for(int u=0;u<m->nv;++u) {
      double inertia=0,centripetal=bias[u]-zero_bias[u];"""
EFFORT_NEW = """    // Validate terms that would otherwise be multiplied by a zero tangent.
    for(int u=0;u<m->nv;++u)if(!std::isfinite(a[u])||!std::isfinite(curve[u])||
        !std::isfinite(bias[u])||!std::isfinite(zero_bias[u]))throw std::runtime_error("inertia: nonfinite projected effort input");
    for(int i=0;i<4;++i)for(int u:support[i]) {
      double inertia=0,centripetal=bias[u]-zero_bias[u];"""


def render_header(source: Path) -> tuple[str, dict]:
    raw = source.read_bytes()
    identity = sha256(raw).hexdigest()
    if identity != BASELINE_SHA256:
        raise ValueError("inertia_baseline_sha256_mismatch")
    code = raw.decode("utf-8")
    for old, new in ((MASS_OLD, MASS_NEW), (EFFORT_OLD, EFFORT_NEW)):
        if code.count(old) != 1:
            raise ValueError("inertia_patch_anchor_count")
        code = code.replace(old, new)
    return code, dict(experiment="exact_zero_tangent_projection_v1",
        baseline_header_sha256=identity,
        candidate_header_sha256=sha256(code.encode()).hexdigest(),
        generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        physics_frequency_changed=False, thresholds_changed=False,
        default_configuration_changed=False, training_eligible=False)
