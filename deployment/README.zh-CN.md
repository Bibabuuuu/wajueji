# 自主挖装项目独立恢复包

本包包含冻结源码、挖机及卡车模型/贴图、机构与显示后端、实际冻结场景配置、参考轨迹及状态、SAC权重、关键原生快照和定制MuJoCo接口。无需原来的项目工作区、Documents运行包或codex-work中的项目资源；Python、CUDA、Newton/Warp、PyTorch/SB3等环境另行准备。

下载入口：[独立恢复包与校验文件](https://github.com/Bibabuuuu/wajueji/releases/tag/portable-restore-20261024)。`main`中的`deployment/`用于查阅和维护工具；以下命令在Release独立包的解压根目录执行。仅`git clone`或下载GitHub自动生成的源码包不包含大型运行资源。

## 恢复和启动

在已配置项目依赖的Linux/NVIDIA环境中，把本包解压到任意新目录：

```bash
tar -xzf wajueji-portable-restore-20261024.tar.gz
cd wajueji-portable-restore
python restore.py check
python restore.py smoke --output ./outputs/first-smoke
```

`check`逐文件验证SHA256；`smoke`加载冻结策略、恢复原生初态并核对24类状态，执行一次策略决策（3个物理帧），同时加载原三维显示模型并生成土壤表面。它不训练，也不运行完整一铲。

以后明确需要重新执行一铲时，使用独立的新输出目录和新预算：

```bash
python restore.py dig --seconds 600 --output ./outputs/new-dig
```

`dig`调用原冻结SAC和原环境，只做推理；本次归档不重新验收完整挖掘或运输效果。它最多7200帧，并保留原保护、预算和失败保存逻辑。输出目录不可重复使用。

## 内容与验证范围

- `repo/`：冻结源码提交`de813c431a42e4977c9238327decec3ee842a168`，补入原运行所需的数据与关键状态。冻结物理源码字节保持不变。
- `resources/frozen-runtime/`：实际使用的旧场景及控制桥输入，保留它们原有的逐哈希绑定。
- `resources/backends/`：机构计算、原显示后端、挖机与卡车资源。
- `restore.py`：只在运行时覆盖资源根目录；不更换策略、控制律、材料或保护阈值。它主动禁止访问打包机上的原项目资源目录。
- `MANIFEST.json`：每个文件的相对路径、大小和SHA256。

最终17页PPT、提交视频和图片位于同一仓库的[final-archive-20261023 Release媒体附件](https://github.com/Bibabuuuu/wajueji/releases/tag/final-archive-20261023)中；历史PPT不作为恢复条件。

历史实验脚本和报告保留原来源路径及旧预算，用于查阅。需要复现其他历史实验时应另定输入与新预算；本包不承诺包含所有旧实验的逐帧粒子回放或训练经验池。
