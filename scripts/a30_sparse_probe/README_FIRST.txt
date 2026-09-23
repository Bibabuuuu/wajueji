A30 47/65石 10cm稀疏网格分段初始化与降频 v3

本包不是APPROACH运行包，不移动铲斗，不入土，不训练，不改正式源检查点。
GPU0使用正式47石，GPU1使用正式65石。两张卡可以分别在两个终端启动。
用户已确认新顺序：120Hz换网格并保存 → 新进程恢复审计 → 从完整120Hz点切60Hz并保存 → 新进程恢复审计。
目标仍为10cm＋60Hz。60Hz输入绑定120Hz候选、独立审计、检查点及源码身份，不接受只保存机构姿态。
每组0.5–2秒网格/频率过渡观察，要求末尾连续0.5秒严格稳定，再测1秒纯静置。
60Hz下MuJoCo仍执行两个原有1/120秒段，各40内部步，同一平均力保持，总冲量不重复。
机构控制仍120Hz。每个候选保存后由新进程立即执行2秒严格保持，不做隐藏预热。
全部过渡帧如实保留；候选通过不表示从首帧起全部严格，也不授予正式数据资格。
120Hz换网格沿用本地成功初始化：有界warmup内p99瞬态不单独中止，但仍算严格失败并清零尾窗。
60Hz降频保留p99>0.05m/s诊断中止，不沿用网格转换的例外。
其余原诊断保护保留：非有限/禁止或铲斗接触、闭链>1mm、石速>0.1m/s、角速>1rad/s、粒速max>0.5m/s、石块相对源位移>5mm立即停止。
严格稳定门完全不变（石速0.01m/s、角速0.15rad/s、粒速max0.05m/s、p99 0.01m/s）；测量及独立恢复从第一帧起均须严格。
5mm位移保护相对于每阶段的完整输入检查点；同时报告相对正式原点的累计位移，不把阶段位移称作总漂移。
v1/v2拒绝结果不改判。失败摘要直接列出具体保护项和帧值。
稀疏候选保存阶段诊断NPZ和元数据；即使恢复通过，也不等于运动、入土或正式采集资格。

预计每张卡3–10分钟；首次内核编译可能更久，不承诺复现本地提速倍数。
启动前确认该卡无其他任务。脚本会检查GPU计算进程并拒绝覆盖运行。

上传 .tar.gz 与 .tar.gz.sha256 后（若不在/home/user01，先定位上传目录）：
  find /home/user01 -maxdepth 3 -type f -name 'a30-layered-sparse-frequency-v3.tar.gz' -print
  cd 实际上传目录
  sha256sum -c a30-layered-sparse-frequency-v3.tar.gz.sha256
  test ! -e /home/user01/a30-layered-sparse-frequency-v3 && tar -xzf a30-layered-sparse-frequency-v3.tar.gz -C /home/user01

GPU0终端：
  bash /home/user01/a30-layered-sparse-frequency-v3/run_47_gpu0.sh

GPU1终端：
  bash /home/user01/a30-layered-sparse-frequency-v3/run_65_gpu1.sh

所有日志/JSON/诊断快照在本包outputs/。不用下载NPZ，只回传COPY_RESULT_BEGIN/END内容。
失败不覆盖旧结果，任何阶段拒绝就停止后续。已通过的120Hz完整点保留，后续60Hz失败不必重做初始化。
仅当同一v3包的120Hz候选和独立审计都通过，才可用确切路径续测60Hz（65石换成自己的两个报告和run_65_gpu1.sh）：
  LAYERED_SPARSE_120_REPORT=/实际路径/120hz.json LAYERED_SPARSE_120_AUDIT=/实际路径/120hz-resume.json bash /home/user01/a30-layered-sparse-frequency-v3/run_47_gpu0.sh
不自动挑最新文件，不接受旧v1/v2报告，不混用47/65石；绑定会在60Hz物理之前重新检查。
本地诊断源结果不能代替A30正式源验收。测速是顺序静置窗口比较，不是同初态接触动作对照。
请回传失败报告，不放宽稳定门、不修改平台源码。原检查点不变，无需重新构造47/65石。
