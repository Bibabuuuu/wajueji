A30 47石前馈优化验证 v1 — 仅GPU0，不运行GPU1/65石

上传压缩包和同名.sha256文件，不上传文件夹。上传后先确认实际目录。
若上传到/home/user01：
cd /home/user01
sha256sum -c a30-layered-47-optimized-motion-v1.tar.gz.sha256
test ! -e /home/user01/a30-layered-47-optimized-motion-v1 && tar -xzf a30-layered-47-optimized-motion-v1.tar.gz -C /home/user01
bash /home/user01/a30-layered-47-optimized-motion-v1/run_47_gpu0.sh

保留旧a30-layered-sparse-frequency-v3目录及outputs。
从其已通过审计的60Hz完整点恢复，不重做场景/网格/降频初始化。
输入SHA：003dafa7557cda74d22f53c5b0282079ae162e2b4cc889c3535662d388152589。
唯一改动是D-107机构前馈精确零项投影；10cm/60Hz、4800Hz内部刚体步、动作和安全门均不变。
依次保持、四轴小幅正向运动与停止，共270帧/4.5秒仿真。预计几分钟，不是长APPROACH或挖掘。
每15帧输出进度，每阶段保存完整诊断点，异常保留报告/日志。点未获独立恢复或训练资格。
脚本检查包清单、旧报告存在、GPU0空闲和任务锁；失败不要删除旧结果或放宽门槛。
本包不改变旧默认入口，不写正式检查点，不采集或训练。结束后回传COPY_RESULT。
