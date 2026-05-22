# 数据集目录

所有 `koopman_*.npz` 与仿真数据集中存放在此。路径常量在 `koopman/paths.py` 中定义。

| 文件 | 说明 |
|------|------|
| `koopman_train_merged.npz` | 默认训练集 |
| `koopman_val.npz` | 验证集 |
| `koopman_test.npz` | 默认测试集 |
| `koopman_train.npz` / `koopman_train_left_turn.npz` | 合并前的子集 |
| `koopman_dataset_v1.npz` | v1 历史数据 |
| `sim_10HZ.npz` | 仿真 10Hz |
| `koopman_test_dataset.npz` | 小规模补充测试（7 段） |

由 rosbag 重新生成时，运行 `scripts/data/` 下工具，输出会写入本目录。
