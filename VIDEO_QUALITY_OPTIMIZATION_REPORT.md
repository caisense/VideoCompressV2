# 视频独占带宽与增强画质优化交付报告

日期：2026-10-16。交付状态：**三档600秒候选验证通过，独立安装和双端一键启动验证通过；不宣称三档1800秒最终稳定性通过。** 用户已明确取消30分钟长测，不再续跑。首轮CAP60/120的失败和CAP180的中断均保留，不能被取消或网络修复改写为PASS。

当前部署 `/opt/atk/video_quality`；发送器SHA256 `cf8b7d0340a877ab36b55a1422f99371c8703fd4880ba0b14b30a5a4b973d081`。配置保持 `CANDIDATE_600S`，属于目前最接近可用、具有明确验证范围的三档候选，而非完整约束下已经认证的最终稳定版本。实验根目录 `runs/video_quality_optimization/hardware_resume_20261013/`，下文证据名均相对此目录。

## A. 实现、环境与兼容性

实际链路：RK3588摄像头 → RGA/RKNN YOLOv8-Seg → ROI → MPP H.265 → 全局应用令牌桶/RTP/UDP → Windows解码 → 有界最新帧GAN → HUD/窗口提交。音频关闭且预算0，音频模块保留，不启动Codec2采集/发送线程。系统后台协议不属于应用令牌桶，因此另采全网卡证据，不能将应用限速当作所有系统流量的保证。

| 路径 | 实际变更或复用 |
|---|---|
| `cpp/common/config.*` | 独立ganq60/120/180、音频off/零预算、参数/对齐/预算边界；旧rate/rebuild/gan2/gan4默认值保留 |
| `cpp/capture/`、`frame_meta.h` | 质量路径采用有界最新V4L2缓冲、驱动单调时钟PTS；单个拥有独立缓冲的RGA任务等待20ms，超时/忙碌/错误显式CPU RGB→NV12恢复，连续不可用超过5s报错；旧路径保留 |
| `cpp/roi/adaptive_rate_controller.h`、`cpp/main.cc` | 面积×权重帧预算近似、有界平滑delta QP、IDR/GOP/队列/全帧编码反馈；按任务所列Meddeb等2014年ROI预算思想近似实现；不是完整CTU独立R–λ控制器，无硬件逐区域比特观测 |
| `cpp/transport/async_rtp_sender.*` | 修正参考链恢复/IDR请求；保留全局应用pacer和有界队列 |
| `tools/live_h265_hud.py` 等 | 实际模型/Provider/精度与尺寸、异步工作器身份、真实唯一输出计数、HOLD/fallback/窗口提交与源帧龄证据 |
| `tools/run/analyze/select_video_quality_*`、验收/网卡工具 | 同源扫描、Pareto候选、严格解码、全网卡pcap/时间/计数校验，不用任意总分掩盖取舍 |
| `tools/start_video_quality.*`、`launch_video_quality_sender.py`、`video_quality_profiles.json` | 三档统一配置、模型/标签哈希、板端本地日志、Windows环境预检、自动x4/x4/x2及可选none |

实际接收GPU为RTX5060Ti 16GB、驱动616.92；Python `D:\Python\Python311\python.exe`，ORT1.30。RTX3060Laptop6GB为NOT RUN，不从显存占用推断其实时性能。模型存储/输入为FP32；请求TensorRT FP16，实际Provider链含TensorRT/CUDA/CPU，HUD和日志明确fallback。已有节点Provider trace，不据此宣称全部算子均为FP16。

虚拟机使用 `/opt/atk-dlrk3588-toolchain`，Codec2源码 `/home/alientek/codec2-1.2.0-goal0901`。18项C++测试与交叉构建通过；当前86个主要模块/测试文件与虚拟机构建源码在仅规范化换行后哈希一致，见final_build_source_consistency.json。真实三档文件源、摄像头、旧gan4和受控延迟共7项功能检查通过；受控阻塞不冒充自然硬件故障或长期验收。原稳定 `/opt/atk/rknn_yolov8_seg_cam` 未替换，隔离实验目录保留v6回滚文件；新独立安装manifest记录实际库和模型。未commit/push；用户既有变更保留。

## B. 三档候选与实测

| 项目 | CAP60 | CAP120 | CAP180 |
|---|---|---|---|
| 编码/解码尺寸 | 320×180 | 320×180 | 384×216 |
| FPS | 6 | 10 | 8 |
| TARGET kbps / 物理CAP kbps | 45 / 60 | 90 / 120 | 155 / 180 |
| GOP秒 | 2 | 4 | 4 |
| P QP / I QP | 20–40 / 28–40 | 20–40 / 24–40 | 16–40 / 24–40 |
| GAN | x4 | x4 | x2 |
| AI原生输出 | 1280×720 | 1280×720 | 768×432 |
| 最终呈现 | 1280×720 | 1280×720 | 1280×720，含传统后缩放 |
| ROI | 固定映射，自适应off | 固定映射，自适应off | 固定映射，自适应off |
| 600秒候选 | PASS | PASS | PASS |
| 首轮1800秒 | FAIL：全链路1s峰值 | FAIL：全链路1s峰值、源帧龄 | SSH中断，未完成 |
| 修复网络后的1800秒 | 用户取消，只有部分采集 | 用户取消，未运行 | 用户取消，未运行 |

固定ROI沿用YOLO物体分割，背景/halo/核心/边缘的名义相对QP为+14/+2/−7/−11，受硬件和QP约束；off不等于没有ROI。CAP120人物测量窗口599/599帧存在非背景ROI，文字573/599帧；文字场景的物体分割不等于文字定位。未接入独立face/OCR模型，固定语义裁剪只用于质量评价。

### 600秒文字参考循环：码率、增强、延迟

以下为真实板到PC、剔除10秒预热、各测量600秒；不是实时摄像头输入。延迟是软件采集PTS至窗口提交，非光学延迟。成功增强延迟排除HOLD/fallback，另验全部图像源帧龄。

| CAP | ES kbps | RTP负载 kbps | 视频wire kbps | 全出站wire kbps | 视频wire/CAP | 解码 / 有效GAN fps | GAN成功率 |
|---|---:|---:|---:|---:|---:|---|---:|
| 60 | 28.69 | 28.48 | 35.52 | 35.90 | 59.20% | 5.995 / 5.575 | 92.99% |
| 120 | 24.15 | 23.81 | 32.82 | 33.20 | 27.35% | 9.988 / 9.072 | 90.82% |
| 180 | 40.53 | 40.30 | 48.69 | 49.07 | 27.05% | 7.992 / 7.977 | 99.81% |

ES包含编码Annex-B数据，RTP负载来自发送计数差、剔除RTP头；实际首尾边界略有差异，不用两列相减推断头开销。视频wire来自UDP5014完整物理开销，全部出站wire还包含ARP/DHCP/SSH等。

| CAP | 全出站1/5/60s峰值 kbps | 成功GAN延迟p50/p95/p99/max ms | 推理p50/p95/p99/max ms | 最大源帧龄ms | 稳态RSS变化MiB / 斜率MiB/min |
|---|---|---|---|---:|---|
| 60 | 68.36 / 51.03 / 37.43 | 274.6 / 337.1 / 433.8 / 473.2 | 78.0 / 81.7 / 84.4 / 99.3 | 626.7 | -15.69 / 0.226 |
| 120 | 118.37 / 67.69 / 35.66 | 201.0 / 214.1 / 220.6 / 302.5 | 79.3 / 82.4 / 86.5 / 99.2 | 436.8 | -16.51 / -2.805 |
| 180 | 173.15 / 104.29 / 52.69 | 195.6 / 247.4 / 314.7 / 397.4 | 42.9 / 45.8 / 49.0 / 57.6 | 459.4 | -16.11 / 0.219 |

| CAP | 队列最大帧数 / 最大年龄ms | IDR比特p50/p95/max | P帧比特p50/p95/max | 全GPU占用采样最大MiB | CPU预处理恢复帧 |
|---|---|---|---|---:|---:|
| 60 | 1 / 0.2 | 20760 / 21424 / 21576 | 3240 / 8104 / 12096 | 2314 | 0 |
| 120 | 2 / 114.6 | 30136 / 30560 / 30640 | 1104 / 5576 / 12824 | 2311 | 4 |
| 180 | 2 / 125.0 | 36584 / 37400 / 40408 | 2360 / 14011 / 24936 | 2118 | 3 |

GPU占用是整卡采样值，含桌面和其他进程，不是GAN独占显存。三档600秒验收均无丢包/乱序/严格解码错误，队列、内存、源帧龄和fallback策略通过；逐帧QP、I/P类型、GOP相位和比特保存在encoder.jsonl，不能从均值推断所有帧。证据 `candidate600_v7_text_summary.json` 与各run的acceptance、原始pcap/日志。

### 同源60秒质量评价

参考为真实640×480采集，重采样6/8/10/12fps的板端/本机文件逐帧像素哈希一致；统一Lanczos变换至1280×720，不是原生720p真值。PSNR/SSIM为灰度，ROI为固定语义裁剪，时间残差不做运动补偿。不同FPS不能当作完全相同帧总体。LPIPS/OCR依赖不可用，均NOT RUN。

| 场景/CAP | PSNR / ROI-PSNR dB | SSIM / ROI-SSIM | 时间残差波动 |
|---|---|---|---:|
| moving_person/60 | 27.296 / 21.890 | 0.88252 / 0.71395 | 1.0537 |
| moving_person/120 | 27.274 / 21.821 | 0.88472 / 0.71634 | 0.6711 |
| moving_person/180 | 28.558 / 22.786 | 0.90175 / 0.74698 | 0.7471 |
| text/60 | 26.354 / 20.087 | 0.87303 / 0.60856 | 1.2977 |
| text/120 | 26.415 / 20.136 | 0.87652 / 0.61847 | 0.8632 |
| text/180 | 28.029 / 21.454 | 0.89751 / 0.67023 | 0.9044 |

三档600秒文字SSIM/ROI-SSIM分别0.87085/0.60464、0.87581/0.61830、0.89707/0.67061，采样窗口不同于60秒，不能直接相减作为版本退化或提升。

## C. 九项结论与限制

1. **CAP60→120→180的变化**：人物SSIM0.88252→0.88472→0.90175，文字0.87303→0.87652→0.89751；ROI见上表。120相对60的收益较小，并非所有指标单调上升；人物PSNR略降。FPS、尺寸和GAN倍率同时变化，是系统方案取舍而非CAP单变量因果证明。
2. **尺寸/FPS选择**：目前GAN路径内候选为320×180@6/x4、320×180@10/x4、384×216@8/x2。384×216@8/x4曾因有效GAN帧率/fallback被拒绝。保留none与其他Pareto路径，未证明全局最优；RTX3060未验证。
3. **分辨率与码率哪个更有效**：B阶段固定QP28–32/GOP2/x4，10fps/TARGET90时256→320使人物SSIM+0.01801、ROI+0.04230，文字+0.02237/+0.08714；固定320@8将TARGET85→100几乎不改变ES或质量。该范围提高输入信息量比仅提高TARGET有效；不是固定实际码率下的纯分辨率实验，也不等于降低QP无效。见controlled_resolution_vs_target_B.json。
4. **x2/x4/none**：v7相同解码帧的普通缩放在整体和ROI-SSIM上经常优于GAN。CAP120人物none/x4为0.90120/0.88472，ROI0.74871/0.71634；文字0.89305/0.87652，ROI0.62784/0.61847。GAN可平滑块状边界、改变轮廓，也会损失/生成细节；不能宣称真实小字或人脸恢复。x4/x4/x2是请求GAN路径内的候选，保留`-Enhancer none`供偏重参考保真的使用者选择。见v7_same_input_none_x2_x4_quality.json。
5. **自适应ROI**：v7两场景12组off/on均通过短测，但收益微小且有取舍。CAP60开启后人物ROI-SSIM−0.00152、文字+0.00067，其余差异更小；不宣称统计显著或普遍更优。默认off，保留固定ROI和Pareto备选，见roi_v7_review.json、shared_E_v7_pareto.json。固定ROI运行证据见fixed_roi_runtime_presence.json。
6. **GOP呼吸**：CAP60同源P最小QP24/20/16的GOP相位SSIM跨度约0.00235/0.00621/0.02704；P16增加码率但人物和文字质量反降，因此未选CAP60 P16。不得归咎于GAN。CAP120 P16整体SSIM较高，但人物ROI与时间残差较差、文字曾超限，保留取舍而非直接推荐。见CAP60_qp_gop_phase_comparison.json、fixed_i_both_scenes_comparison.json。
7. **低于TARGET的原因**：QP范围、I/P分配、内容复杂度与低输入信息量约束了实际编码。TARGET扫描欠填，降低基础QP才明显增加ES；没有填充/噪声。当前ROI相对映射基本沿用旧方案，主要改善来自基础QP/GOP与尺寸/FPS选择，不是专用face/text ROI。
8. **未利用预算**：600秒实际视频wire利用率约59.20%/27.35%/27.05%，仍明显低于80–95%的初始期望。不能以填充流量或画质呼吸换利用率，也不能将低平均值当作短时峰值安全的证明。
9. **优先后续优化**：全链路短时预算（含ARP等控制流量）、接收窗口偶发长提交间隔、RGA/CPU转换外观一致性优先于盲降QP或添加模型。face/OCR可作为后续专门保护能力，当前未实现；更大尺寸及其他GPU需另行验证。

### 与原有gan4 CAP120的比较

旧方案320×180@8fps/x4、TARGET86/GOP2，与新320×180@10fps/x4、TARGET90/GOP4比较：人物ES14.65→20.85kbps（+42.3%）、SSIM0.87873→0.88472、ROI+0.00463；文字ES15.38→23.57（+53.2%）、SSIM0.87053→0.87652、ROI+0.01354。帧数增加25%，平均每帧比特实际增幅约13.9%/22.6%，不能把全部码率增幅解释为每帧画质提升。多参数同时不同，不单独归因于QP/ROI；旧基线缺少全网卡采集，不能升级为全链路验收。16个共同时间点参考帧已逐像素核对一致，并保存2fps采样对照，非实时连续帧率证明。见original_gan4_vs_v7_CAP120.json及visual_original_baseline_v7。

### 30分钟结果、网络修复与取消

首轮CAP60平均全出站35.249kbps，但1s峰值73.696>69.904kbps：视频68.784+DHCP4.912。首轮CAP120峰值135.320>129.904：视频129.736+ARP0.672+DHCP4.912；最大全部图像源帧龄1113.8ms，对应797ms窗口提交间隔，驱动帧龄最大52.1ms、捕获CPU恢复0，根因未确定。其帧序/队列/内存/fallback通过，但整体FAIL。CAP180因SSH会话中断未完成，后续进程已确认终止。

板端eth1同时存在系统静态IP与ConnMan自动寻址；已备份有线settings并将ConnMan设为相同192.168.0.101/24、网关0.0.0.0，其他接口不改。40秒空闲被动采集DHCP0、路由正常。**这不证明所有短时峰值已修复**：例如上述120峰值即使扣除DHCP，视频+ARP仍约130.408kbps，超出判据；验收中不扣除后台流量。新网络条件仅做空闲检查与短启动，未重新完成600/1800秒全链路验证。用户随后取消全部30分钟长测，复测60部分数据保留、120/180未启动，不再续跑。

全链路测量为AF_PACKET/SO_TIMESTAMPNS内核出站观察，逐包pcap与NIC计数校验，物理开销按`max(无FCS以太帧长,60)+24`计入填充/FCS/前导码/IFG，包含所有后台协议；不是外置物理探针。平均须≤CAP，1/5/60s窗口使用原有“CAP加一个1238B包/窗口”的判据，未放宽，不声称每个1s都严格≤CAP。旧数据仅有应用RTP的地方明确区别。

### 可审查的视频与视觉限制

- visual_AB_v7：两场景×三档，共6组连续8秒参考/v6 P24/v7候选，FFV1原件逐帧像素回读验证，另附MP4有损预览。
- visual_original_baseline_v7：两组原始gan4与v7共同时间点2fps采样对照，参考像素一致；另附MP4预览。
- visual_recovery_v7：真实未注入的4帧CPU接管附近连续6秒及per_frame_metrics.csv。4帧CPU的平均SSIM约0.90738、ROI0.77684，其他56帧约0.87843/0.63301；不同时间组不能直接作为纯转换器因果收益。切入/返回残差约3.29，对比RGA帧均值约0.75，说明外观转换有时间代价，不等价于无缝恢复。
- 已查看6组对齐首帧、CAP180两场景0/2/4/6秒采样、恢复切入/返回图及原始基线文字样本。差异主要在轮廓、背景和大字；不证明所有区域更优或小字识别提高。画面可旋转仅用于观看，独立原件保留未标注1280×720。
- 所有上述视频是离线GAN重建，按名义FPS播放，不能证明实时延迟、fallback或吞吐；性能仅用实机日志。指标在无标签数组上计算，不用有损MP4评价。详情visual_quality_review.json。

## D. 一键启动、日志与回滚

先在Windows启动接收端，再在已登录的板端选择相同档位。普通使用只需选60、120或180。

```powershell
# Windows：双击tools/start_video_quality.cmd，然后输入档位；或：
powershell -NoProfile -File tools/start_video_quality.ps1 -Cap 120
# 可选普通缩放：
powershell -NoProfile -File tools/start_video_quality.ps1 -Cap 120 -Enhancer none
# 仅检查配置与模型哈希：
powershell -NoProfile -File tools/start_video_quality.ps1 -Cap 180 -CheckOnly
```

```sh
# RK3588，替换数字即可：
sh /opt/atk/video_quality/start_video_quality.sh 120
# 接收机改变时可显式传IP：
sh /opt/atk/video_quality/start_video_quality.sh 120 192.168.0.100
```

默认UDP5004，接收IP192.168.0.100。板端脚本会校验二进制、YOLO模型与COCO标签，详细日志保存在`/opt/atk/video_quality/logs/`，避免经SSH不断传日志。PC模型为model/RealESRGAN_x2_dynamic.onnx和RealESRGAN_x4plus_dynamic.onnx；哈希在video_quality_profiles.json。Python选择顺序为-Python、VIDEO_QUALITY_PYTHON、本机生成的receiver_runtime.json、PATH中的python，启动打印实际解释器与可用Provider；依赖为已验证环境的OpenCV、NumPy、ORT及CUDA/TensorRT运行库，脚本不自动装包或保存SSH密码。

PC HUD/码流/包/GAN日志位于runs/video_quality_optimization/live_<CAP>_<时间>/，launcher.json保存实际命令，received.runtime.json保存解释器和接收源码哈希。模型存储dtype、执行后端与执行精度分开显示；覆盖倍率或Provider标记CUSTOM_UNVALIDATED。三档发送脚本真实解码24/40/32帧；三档Windows入口25秒检查均观察到正确模型与TensorRT链、成功新GAN画面、严格解码且退出0，仅证明短功能。见oneclick_sender_startup/summary.json和oneclick_receiver_startup/summary.json。

原稳定安装未覆盖；v7实验目录/opt/atk/goal1013_20261013保留rknn_yolov8_seg_cam.rollback_v6，部署manifest与模型/库哈希保存在candidate_install_manifest.json及板端install_manifest.json。系统有线网络修复的原settings备份路径在network_static_repair.json。回到旧业务配置可继续使用原稳定目录和原启动脚本；不删除必要回滚文件。

## E. 测试命令与证据索引

- Ubuntu交叉环境：`cd cpp && bash ./tests/build_roi_tests.sh`，18个原生测试程序，-Wall/-Wextra/-Werror；native_tests_v7.log/.exit证明通过，含profile/180档、零音频预算、码率边界、ROI预算平滑、传输参考链、有界捕获/转换及既有模块回归。
- 构建：`cd cpp && ROI_CODEC2_SOURCE_DIR=/home/alientek/codec2-1.2.0-goal0901 bash ./build-linux.sh`；cross_build_v7.log/.exit、v7_source_manifest.json、deployment_v7_validated.json。
- 实际Windows解释器执行`-m unittest discover -s tools -p test_video_quality.py -v`（14）、`test_verify_video_quality_run.py`（19）、`test_capture_ethernet_wire.py`（3），另有test_full_frame_enhancer.py的24项回归也通过，共60项Python测试；见final_python_*_tests.log及final_python_checks.json。新启动器另经Python编译、PowerShell/Linux语法/实际启动验证；修复PS5引号和Get-FileHash不可用问题。
- B为112个已运行trial、另3个NOT RUN；C74、D24；v7 ROI12组；已有失败、严格解码失败或缺失质量窗口不伪造补全。细节combined_B_summary.json、combined_C_summary.json、combined_D_summary.json、combined_E_v7_summary.json；原始基线目录moving_person_A_verified/text_A_verified。
- 三档600秒：candidate600_v7_text_summary.json及candidate600_v7_text_verified。首轮长测：final1800_v7_camera_verified及CAP60/CAP120_final1800_*诊断。取消复测：final1800_v7_staticnet_retry_20261016、cancelled_30minute_tests.json。
- 网络：network_static_repair.json、network_static_repair_validation.json，只保存相关有线配置。视觉：visual_AB_v7、visual_original_baseline_v7、visual_recovery_v7、visual_review_samples；FFV1是无损证据，MP4仅预览。
- 用户将系统测试范围限定为单人物与文字；没有系统验证多人/镜头平移、原生720p真值、OCR/LPIPS或RTX3060，均不外推。30分钟复测已取消；以上交付不得表述为所有原始性能目标已经达标。
