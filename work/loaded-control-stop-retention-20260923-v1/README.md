# v1记录适配失败

第1帧物理硬门通过，但新增停止检查缺少axis_velocity字段，KeyError退出。失败与完整recovery保留，不授予运行通过。v2只补齐从native状态读取的速度字段，不改物理或候选参数。
