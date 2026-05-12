# OpenArmX 小方块抓取调试复盘：为什么碰到了还是容易掉

这篇文章记录 OpenArmX 在 robosuite / MuJoCo 里抓取小红方块的调试过程。核心现象是：夹爪能碰到方块，甚至能短暂夹住，但纯物理 lift 时容易掉。后来为了稳定演示抓取流程，保留了视觉 centered-grasp 路径；如果要看严格纯物理接触，需要显式使用 `--pure-physics`。

重要结论：

> “接触到了”不等于“形成稳定抓取”；“能演示抓起来”也不等于“纯物理接触已经稳定”。

## 运行命令

默认视觉夹持演示路径：

```bash
conda activate robosuit
cd /home/y/Desktop/openarmx_robosuite_example
python run_openarmx_standalone_viewer.py --debug-state
```

严格纯物理调试路径：

```bash
python run_openarmx_standalone_viewer.py --pure-physics --debug-state
```

## 这版为什么能抓起来

它不是单靠一个参数，而是一组工程改动：

- 给右夹爪添加显式 fingerpad collision box；
- IK 控制点使用 fingerpad center；
- 夹爪不再关死，而是按方块尺寸留接触友好 gap；
- close 后进入 settle 阶段；
- lift 目标渐进 ramp；
- `--no-assisted-lift` 作为旧命令兼容保留，仍然走当前已好的视觉夹持路径；
- `--pure-physics` 才关闭视觉居中，用来看真实接触是否能托住方块。

## 为什么纯物理还是会掉

robosuite 这个 demo 主要是 IK 后直接写 qpos。Isaac 参考里则是 articulation + PD actuator，有 stiffness、damping、effort limit 和 velocity limit。两者接触响应不同。

所以这里纯物理模式下会出现：

- close 后能短暂接触；
- lift 初期接触能维持几帧；
- 方块随后滑到单侧；
- 最终掉落。

这不是完全没碰撞，而是接触质量不足以稳定承载 lift。

## 为什么 assisted lift 看起来可能假

视觉 centered-grasp 会直接改方块 free joint pose，让它跟随夹爪。这样能展示完整抓取流程，但它是 demo aid，不是纯物理成功。如果目标是写博客讲工程调试，这个现象应该诚实说明；严格接触测试要用 `--pure-physics`。

## 具身智能老师视角

遇到“碰到了但抓不起来”，不要马上进入强化学习，也不要只调摩擦。先按顺序查：

1. collision geometry 是否在真实指腹位置；
2. 接触是双侧还是单侧；
3. 夹爪是否关太死，把物体挤偏；
4. close 后有没有 settle；
5. lift 是否跳变；
6. 控制方式是 PD actuator 还是直接写 qpos；
7. assisted carry 是否只是 demo aid；
8. viewer 是否截图确认视觉效果。

这次最重要的教训是：仿真里的“成功抓取”不是画面好看就够，它取决于几何、接触、控制和状态机是否一致。
