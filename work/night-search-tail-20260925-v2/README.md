# 夜间固定补收/抬离尾段v2

保持v1的native参考感知补收、保斗和原物理保护。默认dig_only仅补收及CLEAR_PILE；最佳复验使用unload_validation追加固定ALIGN/DUMP。参数搜索不修改此控制器。

通过fixed_geometry.py复用D-162的静态车体/机构/地面检查，不再构造全场土粒地形图。未测量的bucket_terrain_m显式为null，不伪造净空。每候选绑定自己的完整控制历史、来源和源码SHA。没有删除或冻结物料，下一铲观测不改。

当前为CPU/导入准备，尚未运行物理。完整范围、预算、前置审核与命令见[主说明](../night-parameter-search-20260925-v2/README.md)。
