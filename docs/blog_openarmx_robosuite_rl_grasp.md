# 从 OpenArmX 小方块抓取到强化学习：一次真实的具身智能调试记录

这篇文章记录的是一次很典型的具身智能工程过程：我们不是一上来就训练一个“万能策略”，而是先让仿真环境稳定、让夹爪和物体的交互可解释，再把已经能跑通的专家轨迹转成数据，最后用行为克隆和 DDPG 微调训练一个策略。

项目路径：

```bash
/home/y/Desktop/openarmx_robosuite_example
```

当前成果：

- 使用的是 OpenArmX 双手机械臂模型。
- 当前任务只训练右臂 / 右夹爪抓小方块，左臂保持停放。
- 固定方块位置评估成功率：`5 / 5`。
- 随机小范围方块位置评估成功率：`50 / 50`。
- 训练后的随机位置模型：`checkpoints/openarmx_visual_actor_random.pth`。

## 1. 先讲结论：这不是一步到位的“纯物理抓取”

最重要的工程结论是：

> 当前成功版本是“视觉居中辅助抓取 + 强化学习策略”，不是严格纯物理接触抓取。

这句话非常关键。因为一开始最容易掉进的坑是：看到夹爪已经碰到方块，就以为物理上应该能夹起来。实际上小物体抓取对夹爪碰撞体、摩擦、接触点、控制方式、关节驱动方式都非常敏感。

我们最后保留了两个模式：

```bash
python run_openarmx_standalone_viewer.py --no-assisted-lift --debug-state
```

这个命令是已经调好的视觉居中抓取版本。虽然名字叫 `--no-assisted-lift`，但为了兼容旧命令，它现在仍然走视觉居中抓取。

真正的纯物理调试命令是：

```bash
python run_openarmx_standalone_viewer.py --pure-physics --debug-state
```

纯物理版本目前还不稳定。它适合继续研究碰撞和接触，不适合作为第一阶段强化学习训练目标。

## 2. 为什么“碰上了”还抓不起来

一个有经验的具身智能工程师不会只看画面里“有没有碰上”。真正要问的是：

- 夹爪碰撞体是不是覆盖了真实接触区域？
- 方块是不是刚好卡在两个 fingerpad 中间？
- 夹爪闭合量是不是正确？
- MuJoCo 接触摩擦够不够？
- 控制器是通过 actuator 施力，还是直接写 qpos？
- 物体质量、尺寸、接触 solver 参数是否合理？

这次最关键的一个问题是 OpenArmX 夹爪有两个独立 slide joint。夹爪目标不能把“总开口宽度”同时发给左右两个手指，否则每个手指都按总宽度走，最后实际开口会变成两倍，夹爪看起来闭合了，但仍然夹不紧。

最后修正成：

> 每个手指使用目标总内间隙的一半。

这是一个很小的实现细节，但它决定了方块到底能不能被视觉上夹住。

## 3. 为什么 IsaacLab 看起来更好

我们对比过 IsaacLab 里的 OpenArmX。它看起来更稳定，不代表 robosuite 这边一定错了，而是两边建模方式不同。

IsaacLab 通常加载 USD articulation，并通过隐式 PD actuator 驱动关节。robosuite 这边当前 standalone demo 是自己搭 MuJoCo scene，用 IK 算目标，然后写关节 qpos / ctrl。两边控制链路不同，接触稳定性自然不同。

所以不能简单说：

> Isaac 能抓，所以 robosuite 也应该立刻纯物理抓起来。

更准确的判断是：

> IsaacLab 给了一个参考方向，但 robosuite 这边必须单独把碰撞体、驱动方式和接触逻辑调清楚。

## 4. 工程取舍：先让任务闭环，再追求纯物理

如果一直卡在纯物理接触上，整个强化学习管线就无法开始。这里我采用的是课程式工程路线：

1. 第一阶段：视觉居中抓取，让任务闭环。
2. 第二阶段：采集成功专家数据。
3. 第三阶段：训练策略复现这个成功行为。
4. 第四阶段：加入小范围随机方块位置。
5. 后续阶段：逐步扩大随机范围，最后再回到更真实的纯物理接触。

这不是偷懒，而是具身智能里常用的 curriculum 思想。复杂任务不要一口吃掉，先把最小闭环跑通，再逐步增加真实度和难度。

## 5. 当前环境封装

强化学习环境在：

```bash
openarmx_rl/env.py
```

观测是一个 29 维向量，包含：

- 右臂关节位置：7 维
- 右夹爪关节位置：2 维
- 方块位置：3 维
- fingerpad 中心：3 维
- 方块相对 fingerpad 中心偏差：3 维
- 当前专家目标位置：3 维
- 当前夹爪目标：1 维
- 归一化步数：1 维
- 当前阶段 one-hot：6 维

动作是 4 维：

- `action[0:3]`：对专家目标位置的微调
- `action[3]`：对夹爪开合目标的微调

环境内部仍然使用原来的 staged grasp 状态机：

```text
center_xy -> approach -> descend -> close -> settle -> lift
```

这让策略不是从完全随机动作开始乱试，而是在一个已经可工作的专家轨迹附近学习。

## 6. 数据采集

固定位置采集命令：

```bash
conda activate robosuit
cd /home/y/Desktop/openarmx_robosuite_example

python collect_openarmx_data.py \
  --episodes 20 \
  --max-steps 220
```

随机位置采集命令：

```bash
python collect_openarmx_data.py \
  --output-dir openarmx_visual_grasp_dataset_random_tight \
  --episodes 50 \
  --max-steps 220 \
  --random-cube-pos
```

随机范围是：

```text
x: -0.305 到 -0.270
y: -0.160 到 -0.120
```

为什么范围这么小？因为我们测试过更大的范围：

```text
x: -0.340 到 -0.260
y: -0.180 到 -0.100
```

结果边界位置会失败，尤其是 `x` 接近 `-0.31` 甚至更远时，当前右臂 IK 专家不稳定。对第一版训练来说，喂失败专家数据反而会污染策略。因此先收窄范围，让专家数据干净。

本次最终随机数据采集结果：

```text
50 / 50 success=True
数据目录: openarmx_visual_grasp_dataset_random_tight
```

每个 `.npz` 文件包含：

```text
obs
actions
rewards
next_obs
dones
success
episode_length
```

这个格式参考了 `/home/y/Desktop/UR5e_robosuit_thesis` 里的数据采集和训练风格。

## 7. 强化学习训练

训练代码在：

```bash
train_openarmx_ddpg_bc.py
openarmx_rl/ddpg_bc.py
```

训练方法是 DDPG + BC：

- 先用专家数据做行为克隆，让 actor 学会接近专家动作。
- 再在线交互，用 DDPG 更新 critic 和 actor。
- actor loss 里保留 BC 约束，避免策略一开始偏离太远。

固定位置训练：

```bash
python train_openarmx_ddpg_bc.py \
  --dataset-dir openarmx_visual_grasp_dataset \
  --checkpoint checkpoints/openarmx_visual_actor.pth \
  --pretrain-steps 200 \
  --total-steps 1000
```

随机位置训练：

```bash
python train_openarmx_ddpg_bc.py \
  --dataset-dir openarmx_visual_grasp_dataset_random_tight \
  --checkpoint checkpoints/openarmx_visual_actor_random.pth \
  --pretrain-steps 300 \
  --total-steps 1500 \
  --random-cube-pos
```

本次随机位置训练加载了：

```text
9823 transitions
```

输出模型：

```bash
checkpoints/openarmx_visual_actor_random.pth
```

## 8. 评估结果

固定位置评估：

```bash
python evaluate_openarmx_policy.py \
  --checkpoint checkpoints/openarmx_visual_actor.pth \
  --episodes 5 \
  --max-steps 220
```

结果：

```text
success_rate=1.000
mean_reward=10.063
mean_episode_length=203.0
```

随机位置 10 组评估：

```bash
python evaluate_openarmx_policy.py \
  --checkpoint checkpoints/openarmx_visual_actor_random.pth \
  --episodes 10 \
  --max-steps 220 \
  --random-cube-pos
```

结果：

```text
success_rate=1.000
mean_reward=9.035
mean_episode_length=195.3
```

随机位置 50 组评估：

```bash
python evaluate_openarmx_policy.py \
  --checkpoint checkpoints/openarmx_visual_actor_random.pth \
  --episodes 50 \
  --max-steps 220 \
  --random-cube-pos
```

结果：

```text
success_rate=1.000
mean_reward=9.318
mean_episode_length=196.3
```

这说明在当前小范围随机位置里，策略已经能稳定完成任务。

## 9. 可视化命令

看固定位置策略：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_JIT=1 \
python evaluate_openarmx_policy.py \
  --checkpoint checkpoints/openarmx_visual_actor.pth \
  --episodes 1 \
  --max-steps 220 \
  --render
```

看随机位置策略 10 组：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 NUMBA_DISABLE_JIT=1 \
python evaluate_openarmx_policy.py \
  --checkpoint checkpoints/openarmx_visual_actor_random.pth \
  --episodes 10 \
  --max-steps 220 \
  --random-cube-pos \
  --render
```

注意：MuJoCo 有头 viewer 在某些机器上退出时可能出现：

```text
corrupted double-linked list
```

这发生在 viewer / OpenGL 清理阶段，不代表策略失败。判断策略是否成功，看终端里的：

```text
success=True
success_rate=1.000
```

如果想生成视频而不是开窗口，可以用：

```bash
python record_openarmx_policy_video.py \
  --checkpoint checkpoints/openarmx_visual_actor_random.pth \
  --output openarmx_policy_rollout_random.mp4 \
  --frames-dir openarmx_policy_frames_random \
  --episodes 10 \
  --max-steps 220
```

## 10. 这次最重要的经验

第一，具身智能不是只写 RL 算法。真正耗时间的是把机器人、夹爪、物体、接触和控制链路调到一个可以学习的状态。

第二，看到“夹爪碰到了物体”不代表任务成立。接触是否能产生稳定夹持力，是另一个层面的问题。

第三，不要把 IsaacLab 和 robosuite 的结果做简单一一对应。仿真后端、资产格式、控制器、驱动方式不同，接触表现会不同。

第四，先闭环，再泛化。先固定位置成功，再做小范围随机位置，最后再逐步扩大范围。这比一上来训练大范围随机任务更稳。

第五，强化学习不是替你解决所有工程问题的魔法。它更像是在一个已经合理的任务设计、状态空间、动作空间和奖励结构上继续优化行为。

## 11. 当前局限

当前版本还有几个明确边界：

- 不是纯物理接触抓取。
- 不是双臂协同抓取，只训练了右臂。
- 随机位置范围还比较小。
- 观测是低维状态，不是图像输入。
- 策略依赖视觉居中辅助逻辑，不能直接迁移到真实机器人。
- 当前 PyTorch 是 CPU 版，训练速度不如 GPU。

这些不是失败，而是路线图。一个成熟的具身智能项目应该清楚写出自己的假设和边界。

## 12. 下一步路线

推荐下一步按这个顺序推进：

1. 扩大随机位置范围，例如逐步尝试：

```text
x: -0.310 到 -0.265
y: -0.165 到 -0.115
```

2. 把失败样本单独记录出来，分析失败集中在哪些位置。

3. 改进专家 IK，让更远位置也能稳定到达。

4. 回到纯物理接触问题，继续修 fingerpad collision、摩擦和 actuator 控制。

5. 加入图像观测或相机观测，让策略更接近真实机器人输入。

6. 再考虑 sim-to-real，而不是现在就急着上真机。

这就是这次实验最值得带走的东西：具身智能不是“训练一下就会了”，而是先把世界建对，再把任务拆小，再让策略在可学习的环境里逐步成长。
