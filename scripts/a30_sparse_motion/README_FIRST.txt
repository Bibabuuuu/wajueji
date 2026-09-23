47石10cm稀疏/60Hz四轴非接触小动作与停止短测 v1

只运行GPU0，不启动65石、不入土、不训练，不改变手柄默认值。
直接只读恢复已通过v3审计的60Hz完整点003dafa7…152589，不重复120Hz初始化或60Hz降频。
必须保留/home/user01/a30-layered-sparse-frequency-v3及其outputs原样：输入还绑定其120Hz祖先点/报告。
本包保留v3生产者/验收器的精确源码，验证输入SHA与独立恢复报告后才开始物理。

协议：60Hz先静置0.5秒；回转/动臂/斗杆/铲斗依次正向轻动0.5秒，每次后零命令停止0.5秒。
总计4.5秒仿真/270帧。命令分别0.03rad/s、0.003m/s、0.003m/s、0.006m/s。
全程原岩土严格稳定，零铲斗接触、零禁止接触、原闭链及位移保护，保守垂直净空至少0.15m。
验证每轴有实际响应，并在停止段最后0.25秒满足轴速度及铲斗速度门。
任何失败就停，不放宽门；逐帧日志、每阶段完整诊断NPZ和异常摘要保留。
共最多9个52MB阶段快照，请预留约0.6GB空间；不会覆盖输入及旧输出。
阶段快照未经新的独立恢复验收，不冒充训练或合格恢复点。
通过只代表这四个小幅正向动作与停止窗，不代表反向/全行程、APPROACH或接触挖掘。

上传a30-layered-47-sparse-motion-v1.tar.gz和.tar.gz.sha256到/home/user01，执行：
  cd /home/user01
  sha256sum -c a30-layered-47-sparse-motion-v1.tar.gz.sha256 && test ! -e a30-layered-47-sparse-motion-v1 && tar -xzf a30-layered-47-sparse-motion-v1.tar.gz
  bash /home/user01/a30-layered-47-sparse-motion-v1/run_47_gpu0.sh

若上传不在/home/user01，先用下面命令定位并在实际目录校验：
  find /home/user01 -maxdepth 3 -type f -name 'a30-layered-47-sparse-motion-v1.tar.gz' -print
解压时加-C /home/user01。
启动前检查GPU0没有其他物理任务；脚本会检查占用，不会杀进程。
预计2–4分钟，首次编译可能更久。结束回传COPY_RESULT；无需下载NPZ或重复运行旧频率包。
