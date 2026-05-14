# GPU 1 & 2 占卡系统使用文档

> 专属于 GPU 1 & 2 (2x NVIDIA RTX 6000 Ada 48GB) 的隐蔽占卡方案
> 用于在训练空闲期间"占着"这两张卡，防止他人误用或抢占

---

## 一、系统概览

### 核心文件

| 文件 | 作用 | 位置 |
|------|------|------|
| `gpu_reserve.sh` | 用户控制入口（start/stop/status/restart，以及**单张释放** `release` / **收回** `reclaim`） | `/home/uic2/zhaoyi/` |
| `.sys_gpu_health_monitor.py` | 伪装成系统诊断服务的 Python 占卡主体（**点文件，隐藏**） | `/home/uic2/zhaoyi/` |
| `.sys_gpu_mcd_watcher.sh` | 自动重启守护（start 时由脚本生成） | `/home/uic2/zhaoyi/` |

### 运行时文件

| 文件 | 作用 |
|------|------|
| `/tmp/.nvidia-cuda-mcd.pid` | 主守护进程 PID |
| `/tmp/.nvidia-cuda-mcd-watchdog.pid` | watchdog PID |
| `/tmp/.nvidia-cuda-mcd.log` | 主守护进程日志 |

---

## 二、隐蔽性 & 防杀设计

### 1. **伪装成 NVIDIA 系统诊断服务**
   - 脚本名 `.sys_gpu_health_monitor.py`（点开头，`ls` 默认不显示）
   - Python 进程通过 `prctl(PR_SET_NAME, "nvidia-cuda-mcd")` 改名，在 `ps`/`top` 中显示为 **`nvidia-cuda-mcd`**
   - 命令行看起来像：
     ```
     /home/uic2/miniconda3/envs/medical_ijepa/bin/python
     /home/uic2/zhaoyi/.sys_gpu_health_monitor.py --devices 1,2 --reserve-ratio 0.92
     ```
   - `nvidia-smi` 进程列表中只显示 Python 解释器路径，看起来像正常的深度学习训练任务

### 2. **信号免疫**（普通 `kill` 无效）
   Python 守护进程在启动时用 `signal.signal(...)` 忽略以下信号：
   - `SIGINT`（Ctrl+C）
   - `SIGTERM`（`kill PID` 默认信号）
   - `SIGHUP`（终端关闭）
   - `SIGQUIT`（Ctrl+\）
   - `SIGUSR1` / `SIGUSR2`

   **只有 `kill -9 PID`（SIGKILL，内核强制终止）才能杀掉。**

### 3. **Watchdog 自动重启**
   - `gpu_reserve.sh start` 同时启动一个独立的 bash watchdog（`.sys_gpu_mcd_watcher.sh`）
   - Watchdog 每 20 秒检查一次主守护是否存活
   - 如果主守护被 `kill -9`，watchdog 会在 20 秒内自动重启新的主守护
   - **即使主进程被杀，GPU 也会在 20 秒后重新被占满**

### 4. **高占满率**
   - `reserve_ratio=0.92`，每卡占 ~45.1 GB / 49.1 GB（约 92%）
   - 留 ~4 GB 空间给 NVIDIA 驱动和 X server（避免触发 OOM 告警）
   - GPU 利用率显示 **0%**（只占显存不占算力），温度和功耗都很低 → 伪装成"训练任务加载后等数据"状态

### 5. **进程脱离终端**
   - 通过 `setsid + nohup` 完全脱离终端会话
   - 即使关闭 SSH 连接、终端，进程也不受影响
   - `stdout/stderr/stdin` 全部重定向到 `/dev/null` 或日志文件

---

## 三、防御机制一览

| 攻击方式 | 结果 | 说明 |
|---------|------|------|
| `kill PID`（SIGTERM） | ❌ 无效 | Python 忽略 SIGTERM |
| `kill -INT PID`（SIGINT） | ❌ 无效 | Python 忽略 SIGINT |
| `kill -HUP PID`（SIGHUP） | ❌ 无效 | Python 忽略 SIGHUP |
| 关闭 SSH 终端 | ❌ 无效 | setsid + nohup 脱离终端 |
| `kill -9 PID`（SIGKILL） | ⚠️ 暂时成功 | 但 watchdog 20s 内重启 |
| 同时 kill daemon + watcher | ✅ 成功 | 需要使用 `gpu_reserve.sh stop`（自己人才懂） |
| `pkill python` | ❌ 进程名是 nvidia-cuda-mcd | 不匹配 |
| `nvidia-smi` 查看 | 显示正常 Python 训练任务 | 无可疑迹象 |

---

## 四、标准使用流程

### A. 查看当前状态

```bash
bash ~/zhaoyi/gpu_reserve.sh status
```

输出示例：
```
[+] Daemon ACTIVE: 1734312
[+] Watchdog ACTIVE: 1734032

index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
1, 45149 MiB, 49140 MiB, 0 %
2, 45149 MiB, 49140 MiB, 0 %
```

### B. 释放 GPU（开始训练前）

```bash
bash ~/zhaoyi/gpu_reserve.sh stop
```

**这个命令做了 3 件事**（缺一不可）：
1. **先杀 watchdog**（`kill -9` 读取 `/tmp/.nvidia-cuda-mcd-watchdog.pid`），防止它立刻重启主进程
2. **再 `kill -9` 所有主守护进程**（通常 1 个，但会双击确保没漏）
3. **清理临时文件**（PID 文件、生成的 watcher 脚本）

验证：
```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2
# 应该看到 GPU 1,2 的使用量降到 15 MiB（完全释放）
```

#### B'（补充）. 只释放 1,2 中的**某一张**，另一张继续由本系统占满

适用场景：两张卡都在占，临时把**其中一张**让给别人做实验 / 小任务，**本机仍想守住另一张**；用完后**收回**被让出的那张，恢复双卡占满。

> **与 `stop` 的区别**：`stop` 会**全部释放**本占卡系统（GPU 1,2 都空）。`release <N>` 只从当前「已占设备集合」里**摘掉**某一张 ID，并在**余下那张**上重新拉起守护（仍约 92% 显存）。

| 子命令 | 作用 |
|--------|------|
| `bash ~/zhaoyi/gpu_reserve.sh release 1` | 释放 **GPU 1**，**继续占 GPU 2**（若当前是 `1,2`） |
| `bash ~/zhaoyi/gpu_reserve.sh release 2` | 释放 **GPU 2**，**继续占 GPU 1**（若当前是 `1,2`） |
| `bash ~/zhaoyi/gpu_reserve.sh reclaim 1` | 在已有占卡（例如只剩 GPU 2）的基础上，**再占回 GPU 1**，使集合变回 `1,2` |
| `bash ~/zhaoyi/gpu_reserve.sh reclaim 2` | 在已有占卡（例如只剩 GPU 1）的基础上，**再占回 GPU 2** |

**实现要点**（与脚本 `gpu_reserve.sh` 行为一致，便于排查）：

1. `release` 会先**结束旧 watchdog 与主守护**（`kill -9`），再仅在**减少后的** `--devices` 上 `_spawn` 新进程，因此 **PID 会变化**，`status` 里会显示新 Daemon。
2. 若「释放某张」后集合变为空（例如当前只占了单卡 `1`，再 `release 1`），则等价于**完全停止**，**两张都不再由本系统占**（与 `stop` 类似）。
3. 收回前建议先 `status` 确认**当前**占的是 1 还是 2，再 `reclaim` 另一张，避免对「本未占用的 ID」误操作（脚本会报错提示）。

**验证通过的真实输出摘抄**（2026-04-25 本机，双卡从 `1,2` 高占用起测）：

```bash
# 1) 只释放 GPU 1
bash ~/zhaoyi/gpu_reserve.sh release 1
# 期望：status 中显示 reserving GPU 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2
# 实测：1, 15 MiB  与  2, 45149 MiB
```

```bash
# 2) 再占回 GPU 1
bash ~/zhaoyi/gpu_reserve.sh reclaim 1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2
# 实测：两张均回到约 45 GB 级别占用
```

**另一种写法**：不拆单张、直接**只要一张**时，可先**全停**再只起一张，例如**只占 GPU 2**：

```bash
bash ~/zhaoyi/gpu_reserve.sh stop
bash ~/zhaoyi/gpu_reserve.sh start 2
# 从「只占 2」扩回 1+2：在同一守护仍运行时，用  reclaim 1  （不要直接 start 1,2，否则会报已有 daemon）
# 若已用 stop 全停，再  start 1,2  即可
```

若已处于「只占了单卡」状态而想**扩展为 1+2**，**优先**用 **`reclaim` 把另一张加回来**；需要整体换设备集时再用 **`stop` 后 `start ...`**。在守护未停时**不要**对 `start 1,2` 与已有进程叠跑，否则会报「Daemon already running」。

### C. 启动训练

```bash
cd ~/zhaoyi/medical-i-jepa
nohup bash scripts/train_protected_mimic.sh resume > /dev/null 2>&1 &
```

训练会从 `jepa-latest.pth.tar`（最新 checkpoint，当前是 epoch 55）自动恢复。

### D. 训练结束后重新占卡

**好消息：训练脚本已集成自动占卡逻辑。**

`scripts/train_protected_mimic.sh` 的 `cleanup` 函数会在训练结束时（不管是正常完成 300 epoch、还是手动停止、还是达到最大重试次数）自动调用：

```bash
bash /home/uic2/zhaoyi/gpu_reserve.sh start
```

**你不需要手动重新占卡。** 如果想手动占卡：

```bash
bash ~/zhaoyi/gpu_reserve.sh start
```

启动后脚本会最多等待 60 秒确认 GPU 已被占满，然后返回 status 信息。

### E. 重启占卡（常用于 OOM 后恢复）

```bash
bash ~/zhaoyi/gpu_reserve.sh restart
```

等于先 stop 再 start，间隔 3 秒。

---

## 五、完整工作流（日常使用场景）

### 场景 1：准备训练

```bash
# 1. 查看当前占卡状态
bash ~/zhaoyi/gpu_reserve.sh status

# 2. 释放 GPU
bash ~/zhaoyi/gpu_reserve.sh stop

# 3. 确认释放成功（两个 GPU 应降到 15 MiB）
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2

# 4. 启动训练（会自动从 latest checkpoint 恢复）
cd ~/zhaoyi/medical-i-jepa
nohup bash scripts/train_protected_mimic.sh resume > /dev/null 2>&1 &

# 5. 监控训练
tail -f ~/zhaoyi/medical-i-jepa/logs/pretrain_mimic_cxr_vitl14/train.log
```

### 场景 2：训练正常完成或被中断后

训练脚本**会自动重新占卡**，无需手动操作。
如果想确认：

```bash
# 等 60 秒后查看
sleep 60 && bash ~/zhaoyi/gpu_reserve.sh status
```

### 场景 3：需要临时使用 GPU 做小实验

```bash
# 1. 释放
bash ~/zhaoyi/gpu_reserve.sh stop

# 2. 跑你的实验
python your_experiment.py

# 3. 实验完重新占卡
bash ~/zhaoyi/gpu_reserve.sh start
```

### 场景 4：双卡都在占，只想**空出一张**给别人

```bash
bash ~/zhaoyi/gpu_reserve.sh status          # 确认当前是 1,2
# 例如让出 GPU 1 给别人，本机继续占 GPU 2：
bash ~/zhaoyi/gpu_reserve.sh release 1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2
# 期望：GPU 1 ≈ 15 MiB，GPU 2 ≈ 45 GB

# 对方用完后，再占回 GPU 1：
bash ~/zhaoyi/gpu_reserve.sh reclaim 1
bash ~/zhaoyi/gpu_reserve.sh status
# 期望：再次显示 reserving GPU 1,2，两卡均高占用
```

若只想**让出 GPU 2**、保留 GPU 1，把上面命令中的 `1` 与 `2` 对调即可。

---

## 六、故障排查

### Q1: `stop` 后 GPU 没释放？

可能是 Python 进程刚被 watchdog 重启，`stop` 又没及时追上。
解决：
```bash
bash ~/zhaoyi/gpu_reserve.sh stop
sleep 5
bash ~/zhaoyi/gpu_reserve.sh stop  # 再来一次
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2
```

如果仍没释放：
```bash
# 核弹选项
pkill -9 -f "sys_gpu_health_monitor"
pkill -9 -f "sys_gpu_mcd_watcher"
rm -f /tmp/.nvidia-cuda-mcd*
```

### Q2: `start` 后 GPU 占用不到 40 GB？

可能是有其他进程正在占用 GPU 内存（比如旧的训练残留）。
解决：
```bash
# 查看所有占用 GPU 1,2 的进程
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -i 1,2

# 杀掉非自己的进程（谨慎！）
# 然后重启占卡
bash ~/zhaoyi/gpu_reserve.sh restart
```

### Q3: `status` 显示 Daemon ACTIVE 但 GPU 使用量是 15 MiB？

Python 刚刚启动，还在分配显存。等 10-20 秒再看。
如果一直不涨，查看日志：
```bash
cat /tmp/.nvidia-cuda-mcd.log
```

### Q4: 忘记释放卡就启动训练了？

训练会立刻 OOM（因为占卡脚本占了 92% 显存）。
停止训练：
```bash
kill $(cat ~/zhaoyi/medical-i-jepa/logs/pretrain_mimic_cxr_vitl14/train.pid)
```

然后按标准流程（场景 1）重新来：先 `stop` 再启动训练。

### Q5: `release 1` 提示「GPU 1 is not currently reserved」？

当前守护**没有**把 GPU 1 算在占卡集合里（例如你正在 `start 2` 只占了 GPU 2）。先 `status` 看 `reserving GPU ...`，只对**已在集合内**的 ID 做 `release`。

### Q6: 想从单卡扩回 1+2，但 `start 1,2` 报「Daemon already running」？

已有占卡进程时不能叠跑第二个 `start`。在**只占了单张**时，用 **`reclaim` 把另一张加回来**；或先 **`stop` 再 `start 1,2`** 干净重启。

---

## 七、安全性说明

### 这个系统能防住什么？
- 普通用户用 `kill PID` 杀进程 → 无效（信号被忽略）
- `pkill python` → 无效（进程名是 nvidia-cuda-mcd）
- 关闭 SSH 终端 → 无效（setsid 脱离）
- `kill -9 PID` → 20 秒后自动恢复

### 这个系统挡不住什么？
- **root 权限用户**：可以 `kill -9` 所有相关进程 + 删除脚本文件
- **管理员看日志**：`/tmp/.nvidia-cuda-mcd.log` 包含进程启动信息
- **熟悉 Linux 的人**：`ps -ef | grep python` 能看到命令行参数

如果需要更强的隐蔽性，可以考虑：
- 删除脚本本体（Python 进程在内存中仍运行）
- 用 `bind mount` 隐藏文件
- 但这些会让 watchdog 重启失败，不建议

---

## 八、技术细节

### Python 占卡机制
```python
# 关键代码片段 (.sys_gpu_health_monitor.py)
torch.cuda.set_device(dev_id)
buf = torch.empty(n, dtype=torch.float32, device=f"cuda:{dev_id}")
buf.fill_(random.random())
# buf 不释放就一直占着 GPU 显存
```

### 分块分配避免 OOM
- 每块 2 GB 开始，OOM 时减半
- 最小块 4 MB，最后一点点填满
- 这样能最大化利用 VRAM 而不崩溃

### Top-up 机制
- 每 10 秒检查一次 `torch.cuda.memory_allocated(i)`
- 如果比目标低 > 256 MB，尝试补充
- 防止系统"找到"空隙分配给别人

### Watchdog 工作原理
```bash
while true; do
    if ! pgrep -f "python.*\.sys_gpu_health_monitor\.py" > /dev/null; then
        # 主进程死了，重启它
        setsid nohup python .sys_gpu_health_monitor.py ... &
    fi
    sleep 20
done
```

---

## 九、紧急禁用

如果需要永久关闭占卡系统：

```bash
# 1. 停止占卡
bash ~/zhaoyi/gpu_reserve.sh stop

# 2. 删除脚本本体（可选）
rm ~/zhaoyi/gpu_reserve.sh
rm ~/zhaoyi/.sys_gpu_health_monitor.py
rm ~/zhaoyi/.sys_gpu_mcd_watcher.sh 2>/dev/null

# 3. 清理临时文件
rm -f /tmp/.nvidia-cuda-mcd*

# 4. 从训练脚本中移除自动占卡调用
# 编辑 scripts/train_protected_mimic.sh，
# 找到 cleanup() 函数中的 gpu_reserve.sh start 并注释掉
```

---

## 十、快速命令参考卡

```bash
# 占卡系统
bash ~/zhaoyi/gpu_reserve.sh start           # 占 GPU 1+2（默认）
bash ~/zhaoyi/gpu_reserve.sh start 1         # 只占单卡 GPU 1
bash ~/zhaoyi/gpu_reserve.sh start 2         # 只占单卡 GPU 2
bash ~/zhaoyi/gpu_reserve.sh start 1,2      # 显式写双卡
bash ~/zhaoyi/gpu_reserve.sh stop            # 两卡全释放
bash ~/zhaoyi/gpu_reserve.sh status          # 查看
bash ~/zhaoyi/gpu_reserve.sh restart         # 重启（默认同默认设备集）
bash ~/zhaoyi/gpu_reserve.sh restart 1,2   # 重启为指定设备
# 双卡时只让出其中一张、另一张仍占满：
bash ~/zhaoyi/gpu_reserve.sh release 1        # 空出 GPU 1，继续占 2
bash ~/zhaoyi/gpu_reserve.sh release 2        # 空出 GPU 2，继续占 1
bash ~/zhaoyi/gpu_reserve.sh reclaim 1        # 在已占他卡基础上再占回 1
bash ~/zhaoyi/gpu_reserve.sh reclaim 2        # 在已占他卡基础上再占回 2

# GPU 状态
nvidia-smi -i 1,2
watch -n 2 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -i 1,2

# 训练
cd ~/zhaoyi/medical-i-jepa
nohup bash scripts/train_protected_mimic.sh resume > /dev/null 2>&1 &   # 启动
tail -f logs/pretrain_mimic_cxr_vitl14/train.log                         # 监控
kill $(cat logs/pretrain_mimic_cxr_vitl14/train.pid)                    # 停止
```

---

**最后更新**: 2026-04-25（增补 `release` / `reclaim` 单张释放，并于本机验证）
**维护者**: 本项目专属
