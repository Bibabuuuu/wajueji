# A30离线算力平台运行手册

- 状态：`active-runbook`
- 权威范围：A30平台环境、上传、不可变运行包、双GPU分工、监控和结果回传
- 最后复核：2026-09-22（Asia/Shanghai）
- 阅读条件：准备、启动、诊断或审计A30任务时
- 更新触发：平台环境、驱动、运行时、文件传输方式或运行协议变化

## 1. 平台与本地边界

- 本地唯一开发/Git事实来源是`/home/qunwang/wajueji`。
- 平台主机实测为`znkz-3008`，用户`user01`，Ubuntu 22.04，2张NVIDIA A30 24GB，驱动550.127.05。
- 平台不能访问外网；只能上传文件，不能依赖`pip install`、`git clone`或在线FetchContent。
- 平台文件持久保存，网页/SSH断开不等于进程停止。
- 正式源码修复必须回到本地生成新版本包，禁止只在平台热改后不回填。
- 本地路径`/home/qunwang/...`和平台路径`/home/user01/...`不可混用。运行时读取的包内资源不得引用打包机绝对路径。

## 2. 固定运行环境

- Python/Newton/Warp：`~/newton-warp-offline-py312-a30-v2/run_python.sh`
- Python 3.12.14、NumPy 2.5.3、Warp 1.17.0、Newton 1.5.1。
- 刚体侧使用项目定制MuJoCo 2.3.8；平台兼容库SHA256：`1d2184dad86712e245b5169437c294c1b42580247bc0f8bfb75857360aa7fee2`。
- 系统Python不能假定包含项目依赖。

## 3. 上传包

- 网页传输只能上传文件；目录必须先打包为新的不可变`<name>-vN.tar.gz`。
- 同时生成外层`<name>.tar.gz.sha256`，包内保留逐文件`MANIFEST.sha256`。
- 每次修订使用新目录和新SHA，旧包明确停用；不得覆盖旧源码、日志、检查点或结果。
- 上传位置由文件传输面板当前目录决定，不要假定一定是`/home/user01`。上传后先定位：

```bash
find /home/user01 -maxdepth 3 -type f -name '<包名>.tar.gz' -print
```

- shell通配符`*`前不要添加反斜杠。
- 校验和解压示例：

```bash
cd <上传目录>
sha256sum -c <包名>.tar.gz.sha256
tar -xzf <包名>.tar.gz -C /home/user01
cd /home/user01/<包目录>
sha256sum -c MANIFEST.sha256
```

## 4. 便携性硬门

- 活跃脚本、配置、源码和生成模型不得读取`/home/qunwang`、旧工作树或本地依赖路径。
- 包内资产使用相对包根/仓库根路径；平台固定检查点和固定离线运行时可以作为显式白名单。
- 在昂贵MPM之前执行：清单校验、配置绑定、检查点身份、模型生成、桥编译、MuJoCo加载和READY。
- 必须从任意新解压目录做便携复核，不能只在原打包目录测试。
- 路径或配置预检失败不是物理失败；应说明是否读取了源检查点、是否启动GPU物理、是否写出任何状态。

## 5. 双GPU规则

- 一个连续物理实例固定在一张明确GPU；没有证据表明同一MPM状态可跨两张A30线性加速。
- 第二张卡用于独立审计、独立复跑、不同候选或只读分析。
- 不在同一GPU并发两个连续物理任务。
- 启动前同时检查进程和GPU：

```bash
pgrep -af 'run_|bridge|python'
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

- 单次GPU利用率为0但进程存在，可能是CPU/GPU耦合等待；不能仅凭一帧`nvidia-smi`中断。

## 6. 长任务与进度

- Warp首次运行会编译内核，随后可能长时间静默。静默不等于卡死。
- 长任务应周期输出阶段、帧数、稳定窗、最大速度、首个失败项和有限值。
- 不进行超过60秒的阻塞等待；可使用第二终端观察进程和日志。
- 失败运行使用新的运行ID；不覆盖原日志或报告。
- 运行中不要修改对应解压目录、切换分支或替换源码。

## 7. 检查点与路径

- 检查点必须通过schema、status、来源配置/运行器SHA、Newton/Warp版本、粒子/石块数、solver状态声明和文件SHA验证。
- 同目录可能同时存在正式检查点和`diagnostic_only`文件；必须显式给出完整路径，不能选择“最新文件”。
- 源检查点只读消费；运行前后核对SHA、大小和mtime。
- 平台路径必须写成`/home/user01/...`；本地文档链接使用`/home/qunwang/wajueji/...`，二者不能互相替代。

## 8. 输出与回传

- 每个入口结束时输出`COPY_RESULT_BEGIN`/`COPY_RESULT_END`小型JSON。
- 至少包含状态、退出码、报告是否存在、输入/输出路径及SHA、关键计数、检查、墙钟、首个失败和下一道门。
- 不把Traceback、日志或COPY_RESULT粘回shell执行。
- 平台大数组和长日志留在平台；需要进一步分析时上传新的只读分析脚本，就地生成精简摘要。
- 画面通过VNC/截图复核；录屏不能把不同运行拼成同一连续三铲。

## 9. 失败分类

- 清单/路径/依赖/配置路由失败：便携或入口失败，不算物理失败。
- 检查点身份/恢复失败：续步合同失败，不算新场景动力学失败。
- 非有限、边界、接触所有权、动量或稳定门失败：物理/资格失败。
- 报告缺失或进程中断：运行失败，不能从已有片段推导通过。
- 修复后使用新版本名和SHA，并在[CURRENT_STATUS](../CURRENT_STATUS.md)、[验收](../acceptance.md)或[决策](../decisions.md)记录边界。
