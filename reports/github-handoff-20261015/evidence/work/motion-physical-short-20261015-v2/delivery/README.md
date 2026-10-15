# 运输减振与卸料后合并接近：物理短检

**本轮物理已结束、预算关闭。** 状态both_short_cases_completed；loaded=completed、887帧；return=completed、2752帧。总物理3639帧，原生实例2个，推理/训练均0。停止原因None。记录审计True，27/27项；记录完整与两组完成分别判断。卸料后到原参考预备点实际45.8667仿真秒，旧147.0833秒；这是同合格卸料快照及原连续首3帧门后的单例动作时间，不是实机或GPU墙钟吞吐。 举升最大斗姿对参考误差4.11005度→4.00079度，仅小幅改善，减振未解决；回转仅120帧入口，不能外推整段回转。实际指标：{"loaded": {"command_max_error": 0.0, "load_bias_max_error": 0.0, "phase_max_pitch_error_deg": {"CLEAR_PILE": 4.00078526761825, "ALIGN": 1.4383400104547381}, "phase_samples": {"CLEAR_PILE": 764, "ALIGN": 120}}, "return": {"command_max_error": 0.0, "load_bias_max_error": 0.0, "phase_max_pitch_error_deg": {"RETURN_EXIT": 0.01744280562491638, "RETURN_COORDINATED": 0.03544522623670783, "RETURN_REFERENCE": 0.009783527170723757}, "phase_samples": {"RETURN_EXIT": 612, "RETURN_COORDINATED": 1386, "RETURN_REFERENCE": 751}}}。

仅独立短段，未挖掘或做完整两铲；斗腔粒子队列只是保料诊断，不是标定满斗率或无撒料证明。各原保护及CPU路径沿用，失败不自动重试，旧预算不重开。证据：runs/motion-physical-short-20261015-v2/report.json、read-only-audit.json、summary.json、各组display/；work/motion-physical-short-20261015-v2/closure.json。原运行、模型和正式控制/物理源码保留，无Git提交。


源代码在source/，原生恢复及原连续首3帧核验见report.json。全帧粒子/机构缓存留台式机各组display/，不把两组拼成同一连续流程。下一步只根据本次真实结果决定，未批准新的训练、完整挖掘或额外物理预算。
