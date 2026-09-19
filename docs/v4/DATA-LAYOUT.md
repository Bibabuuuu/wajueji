# 本地数据布局

完整四轴工作包保存在：

```text
local-data/archive/v4-full-2026-09-19/
```

该目录位于仓库工作树内部，因此本地开发只需要进入 `/home/qunwang/wajueji`；同时它被 `.gitignore` 和 `.git/info/exclude` 排除，不会随普通 `git add` 上传。

验证完整归档：

```bash
cd /home/qunwang/wajueji/local-data/archive/v4-full-2026-09-19
sha256sum -c /home/qunwang/wajueji/manifests/v4-full-2026-09-19.sha256
```

`03-data/holdout/` 是封存盲测集。任何训练脚本、标准化统计或超参数选择都不得读取该目录。后续运行统一写入仓库根目录的 `runs/`，每次使用独立时间戳目录。
