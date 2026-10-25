"""Select an original initialization path; never rewrite source schema."""
def open_runtime(source, output, protocol, teacher_report):
    if protocol['source_schema'] == 'layered_47_cycle_checkpoint_v1':
        from layered_47_cycle_runtime import CycleRuntime
        return CycleRuntime(source, output, allow_training_reset=True)
    if protocol['source_schema'] == 'excavation_layered_47_sparse_approach_handoff_checkpoint_v1':
        from approach_runtime import ApproachRuntime
        return ApproachRuntime(source, output, teacher_report)
    raise ValueError('unsupported_teacher_source_schema')
