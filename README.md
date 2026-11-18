# 按比例缩小的整脑脉冲模型（只用神经元和突触）

目标：完全照着大脑的结构搭一个模型，先让它能运行、能学习，再去理解它的本质。
做法：每一部分都照一篇读过的论文搭，先单独重现论文的结果，再接进整脑；不是论文里的东西都单独标出。

## 先看哪里

| 文件 | 内容 |
|---|---|
| `MODEL.md` | 模型现状：搭了什么、和论文对上没有、学习部分的全部实验结果、线路核对表、还缺什么 |
| `DESIGN.md` | 整个闭环的纸面设计：七条环各做什么、想法怎样变成肌肉的指令、和"预测下一个词"的对应、搭的顺序 |
| `IDEAS.md` | 设计者的想法和假设（原话）、它们和实验 / 论文的对应、还没验证的部分 |
| `NOTES.md` | 读过的每篇论文说了什么（标明读了哪些章节） |
| `FRAMEWORK.md` | 最初的证据框架（每条标明是实测、综述、模型、假说还是我的推断） |

## 代码

| 文件 | 内容 |
|---|---|
| `sim.py` `fast.py` | 脉冲网络模拟器（numpy 参考实现 + numba 编译的循环），含可变突触 |
| `cortex.py` `cerebellum.py` `bg.py` `wm.py` `space.py` `brain.py` | 皮层、小脑、基底节、工作记忆回路、空间坐标、整脑组装 |
| `run_brain.py` `check_*.py` | 整脑运行；各部分单独和论文对照 |
| `zenke.py` `run_zenke.py` `zenke_ana.py` | 重现 Zenke, Agnes & Gerstner 2015：学出细胞群并保持活动 |
| `check_sequence.py` `check_replay.py` `check_concept.py` | 在上面那个网络里测"学顺序"和"概念细胞怎么形成" |
| `check_concept3.py` `check_plan_go.py` | 两级网络里的概念细胞；"计划 × 放行"的测量 |
| `check_voice.py` `check_voice_look.py` | 耳朵 → 循环层 → 运动细胞 → 嘴 → 耳朵的闭环发声测试；在它保存的状态上不训练直接量 |
| `rate_units.py` `check_rate_units.py` | 同一个网络不用脉冲（每个细胞一个放电率，输入—输出曲线从脉冲网络量出）及其与脉冲版本的对照 |
| `maes.py` `run_maes.py` | 重现 Maes, Barahona & Clopath 2020：循环网络学会并自己回放顺序 |
| `area_learn.py` `check_area_learning.py` | 把学习规则放进一个分层皮层区的试验 |
| `tools/` | 查论文、抽表格、取图谱坐标的小工具；`fetch_sources.sh` 下载运行所需的第三方代码和数据 |
| `data/` | 从论文抄下来的参数表、图谱坐标、各次运行的记录（文本） |

## 运行

需要 python3、numpy、numba。权重、放电记录等大文件没有放进库里，按下面的命令可以重新生成。

```
sh tools/fetch_sources.sh          # 取作者公开的代码和输入图样（重现两篇论文时要用）
python3 check_column.py 1.0        # 皮层柱（Potjans & Diesmann 2014）
python3 run_brain.py               # 整脑
python3 run_zenke.py p10           # 学习网络三阶段（约 1 小时）
python3 run_maes.py                # 顺序网络（约 4 小时）
```
