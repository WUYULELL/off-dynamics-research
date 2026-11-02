# V9阶段0：CPU只读诊断报告

**阶段0已完成并停止。0训练、0优化更新、0模拟器步、0新oracle标签；阶段1未启动。**

本阶段由用户在2026-10-30明确授权。仅读取既有数据、XML、代码、原始TB/episode日志和缓存文件；所有新代码及报告位于新目录。
五项结论：Hopper形态定义匹配OTDF附录；两个目标集均为7个完整边界段加1个EOF截断段；Hopper205持续偏低、206波动且略有回落；R12的T完全复用R11；所需Hopper全量oracle缓存尚不存在。

## 1. 任务定义与归一化核对

论文附录C.4（PDF第23页）的Hopper-morph头部几何与本地XML逐项匹配：

|字段|论文|本地|核对|
|---|---|---|---|
|geom|torso_geom|torso_geom|一致|
|size|0.125|0.125|一致|
|fromto|0 0 1.45 0 0 1.05|0 0 1.45 0 0 1.05|一致|
|type / friction|capsule / 0.9|capsule / 0.9|一致|

本地timestep=0.002，XML SHA256与R14冻结值一致。本次未发现会使R14不能与论文直接比较的头部尺寸偏差；这一核对并不把不同学习器、超参数和评估协议宣称为完整论文复现。
归一化统一由本地OTDF/info.py读取：(原始episode回报−min)/(max−min)×100，不截断分数。Walker min=10.079455055289959、max=4398.43；Hopper min=−26.3360015397715、max=3152.75。

来源：[OTDF正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4ce6738e84876aa79f13c8bc8b7c5eb-Paper-Conference.pdf)，附录C.4与表4；本地原文与渲染证据见sources/OTDF_page_22.png、OTDF_page_23.png。
本地PDF SHA256：`c63103202cd3f8d992d30d448d2f1e35fe0ad573201ced01a5c0cb6409f20b43`。

## 2. 目标数据轨迹与回报

**两个5000行HDF5均无timeouts字段。** 因而没有把缺失字段假装成观测到的timeout。先保存按原terminal标记分段的原始视图；再单列由explicit_next[i]与observations[i+1]不连续恢复的reset边界。
Walker的额外不连续在0-based行2257、3724、4724，恢复后各对应1000步段；阈值0、1e−6、1e−5、1e−4给出完全相同边界。Hopper的所有不连续均与terminal重合。原数据与terminal语义从未改写。
下表为恢复后的逐段结果。完整边界段指结束于terminal或推断的1000步reset；文件开头视为数据起点，未另存reset元数据，不能额外证明其之前没有缺失前缀。末尾段明确不完整，其partial return不能当作完整episode回报。

|任务|段|原始行范围（含端点）|长度|原始回报|归一化回报|结束证据|
|---|---:|---|---:|---:|---:|---|
|walker2d-morph|1|0–326|327|1002.45329|22.61382|terminal|
|walker2d-morph|2|327–589|263|772.81884|17.38100|terminal|
|walker2d-morph|3|590–1257|668|2246.20178|50.95587|terminal|
|walker2d-morph|4|1258–2257|1000|2803.77841|63.66171|推断1000步reset（非原始timeout列）|
|walker2d-morph|5|2258–2724|467|1602.44498|36.28620|terminal|
|walker2d-morph|6|2725–3724|1000|2877.57248|65.34330|推断1000步reset（非原始timeout列）|
|walker2d-morph|7|3725–4724|1000|2790.34001|63.35548|推断1000步reset（非原始timeout列）|
|walker2d-morph|8|4725–4999|275|765.92974|17.22402|EOF不完整|
|hopper-morph|1|0–651|652|1980.05525|63.11220|terminal|
|hopper-morph|2|652–1319|668|2039.79151|64.99124|terminal|
|hopper-morph|3|1320–1962|643|1946.72866|62.06390|terminal|
|hopper-morph|4|1963–2617|655|1994.24687|63.55861|terminal|
|hopper-morph|5|2618–3264|647|1961.11385|62.51639|terminal|
|hopper-morph|6|3265–3910|646|1956.89867|62.38380|terminal|
|hopper-morph|7|3911–4563|653|1983.50860|63.22083|terminal|
|hopper-morph|8|4564–4999|436|1179.77294|37.93886|EOF不完整|

按条数取ceil(n×比例)、至少1条；按回报降序，相同回报保持原段顺序。包括尾段与剔除尾段两种汇总均保留。以下为归一化回报：

|任务/统计集合|n|均值|中位数|最小|最大|样本SD|Q1|前25%均值（条数）|前50%均值（条数）|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|walker2d-morph / 完整边界段|7|45.65677|50.95587|17.38100|65.34330|20.29518|29.45001|64.50250（2）|60.82909（4）|
|walker2d-morph / 包括EOF片段|8|42.10268|43.62104|17.22402|65.34330|21.30974|21.30562|64.50250（2）|60.82909（4）|
|hopper-morph / 完整边界段|7|63.12100|63.11220|62.06390|64.99124|0.97689|62.45010|64.27493（2）|63.72072（4）|
|hopper-morph / 包括EOF片段|8|59.97323|62.81430|37.93886|64.99124|8.94905|62.30382|64.27493（2）|63.72072（4）|

完整边界段的原始回报汇总：

|任务|均值|中位数|最小|最大|前25%均值|前50%均值|
|---|---:|---:|---:|---:|---:|---:|
|walker2d-morph|2013.65854|2246.20178|772.81884|2877.57248|2840.67545|2679.47317|
|hopper-morph|1980.33477|1980.05525|1946.72866|2039.79151|2017.01919|1999.40056|

Hopper完整7段平均原始回报1980.33477，对应63.12100；论文表4为1980.33，换算63.12085，吻合。含436步末尾片段的8段平均仅59.97323，不能与论文完整轨迹均值混用。
Walker论文表4平均2013.11，本地完整段2013.65854；差约0.55原始回报（约0.0125归一化分），不将其称为逐值完全相同。这里报告实际本地文件统计，不修改数据或归一化常数。
完整轨迹均值只是既有行为数据的描述，不能证明BC可达到该分数或证明critic已是瓶颈。阶段3未执行，不代替其判定。
**阶段3尚未确定使用n=7完整段还是包括末尾片段的n=8。** 本阶段只提供两种账本，不擅自修改后续训练集规则；两种n均小于20，若按现有定义均不触发%BC-10。

## 3. Hopper T训练曲线：205、206为何偏低

已从R14各seed的50个评估点、每点10回合原始回报重算，并逐点与TensorBoard交叉核对。没有平滑，没有重跑，没有据峰值挑检查点。峰值与滚动均值仅作事后曲线描述。

![Hopper全部T曲线](figures/hopper_T_all_seeds_v3.png)

|seed|10–100k均值|110–200k|210–300k|310–400k|410–500k|Last5|Final|单点最大值/步数|最高连续5点均值/窗口|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|205|9.56778|10.90048|13.11100|17.96106|20.42418|19.23540|20.46828|37.30903 / 420k|24.19101 / 380–420k|
|206|13.82982|27.29724|22.99517|26.86530|24.97952|25.30323|25.00635|36.86689 / 200k|30.88640 / 180–220k|

**205：主要是一直相对偏低并缓慢改善。** 420k单点37.31后回落，但最高连续5点评估均值仅24.19，最后19.24；不能把一个短暂峰值写成先稳定学到高水平再崩溃。
**206：早期提升后在较低平台波动，存在一定回落。** 180–220k最高连续5点30.89，Last5为25.30，回落5.58；它不是严格单调、也非完全没有回落，但曲线不支持从稳定高分突然全面崩溃的叙述。
上述是描述性读图，不新增正式“崩溃”阈值或显著性检验；10seed全部保留，不能删除205、206来改善T或source组结果。

## 4. 已有T结果、复用关系与配置

主指标为460/470/480/490/500k五点评估均值；SD为训练seed样本SD(ddof1)，Q1线性25%分位。Final仅辅助。

|组别|seeds|Last5均值|样本SD|最差分数/seed|Q1|Final均值|Final SD|Final最差/seed|Final Q1|
|---|---|---:|---:|---|---:|---:|---:|---|---:|
|R11_Walker_T|101,102,103|41.63430|2.77985|39.59374 / 101|40.05124|41.14829|1.15117|40.25375 / 101|40.49891|
|R12_Walker_T_reuse_R11|101,102,103|41.63430|2.77985|39.59374 / 101|40.05124|41.14829|1.15117|40.25375 / 101|40.49891|
|R13_Walker_T|201,202,203,204,205|38.20863|6.74023|27.90896 / 204|35.03502|40.24093|10.89617|21.48773 / 204|39.82914|
|R14_Hopper_T|201,202,203,204,205,206,207,208,209,210|42.53079|13.19871|19.23540 / 205|39.91013|43.56171|13.24517|20.46828 / 205|40.52723|

**R12没有新T实验：它复用R11的同3个文件（seeds101–103），不能算另一批独立证据。R13为新seed201–205，R14 Walker又复用同一R13 T5。Hopper T为R14独立201–210。** R11与R13不同seed，不作同seed配对差；本阶段也未新增训练条件，配对均值/SE/正号不适用。

|组别|原始出处|
|---|---|
|R11 Walker T|oracle_diag/results/coverage_oracle_20261016_r11/runs/tune_T_t10_h500000_s101…103；选择凭据coverage_oracle_20261016_r11_resume2/T_selected.json|
|R12 Walker T|coverage_oracle_20261018_r12/T_selected.json，逐项指向上述R11路径|
|R13 Walker T|coverage_oracle_20261021_r13/runs/confirm_T_t10_h500000_s201…205；finish1复用收据确认|
|R14 Hopper T|coverage_raw_20261026_r14/runs/hopp_w0_s201…210|

逐seed全部数值、绝对路径、audit与评估hash见T_history_per_seed.csv和T_HISTORY.json。

|配置项|R11/R12/R13 Walker T|R14 Hopper T|
|---|---|---|
|温度/更新/actor cosine周期|10 / 500k / 500k|10 / 500k / 500k|
|批大小/通道|64 target，0 source；V/Q/actor均无source|相同|
|actor advantage / target Q|pre-V target advantage / 每梯度更新|相同|
|γ / expectile / τ|.99 / .7 / .005|相同|
|actor/critic学习率|均3e−4|相同|
|观测/奖励|原始观测，无新归一化，reward scale1|相同|
|任务维度|17/6|11/3|

旧T配置中source_coefficient虽存为1，但source batch=0且三个source通道均关闭，没有source损失；R14 T存为0。该无效占位值差异已单列，不把旧T错当作w=1混合组。阶段0没有创建需与R14配对的新训练组，也没有声称跨任务配对。

## 5. Hopper oracle缓存状态与方向

在整个项目（排除.git、Python环境/依赖目录与本次新输出）搜索标准NPY缓存，并检查全部63个NPZ的数组头。未找到999998×11行的完整Hopper-morph源侧oracle缓存及通过验收的完成凭据。
本次所需方向为：**正常Hopper源数据999998 canonical行 → Hopper-morph目标动力学标签**。

|现有文件/凭据|实际内容|能否替代所需全量缓存|
|---|---|---|
|v3/source_oracle1|999322×17，Walker-morph全量缓存|不能，任务不同|
|v3/stage_a2/hopper_source_acceptance.json|正常Hopper source的5000点事实回放验收|不能，未生成morph反事实标签|
|v3/stage_a1/hopper_source_raw_indices.npy|999998行canonical原始索引映射|不能，仅索引|
|v2/stage_c/hopper_morph_source_oracle_fixed.npz|5000个Hopper-morph目标数据状态/动作，反事实回放到正常Hopper|不能，输入集合与动力学方向均不同|
|v3/stage_a1/hopper_morph_target_replay.npz|5000行morph目标数据的事实回放|不能，是target数据验收|

方向核验同时依据source_state/source_action与morph目标HDF5逐位相同，以及v2/stage_c.py、v2/support.py、verify_replay_v2.py与common.make_target_env的代码路径；不以文件名中的source/oracle字样判断。本阶段未创建、修复或补齐任何缓存。

## 6. 复核、异常、命令与下一阶段

主诊断脚本耗时36.50秒；文献联网核查、PDF只读渲染、绘图和人工复核另计。新训练/优化更新/模拟器步/新oracle行均为0。
异常与偏离：目标文件缺少timeouts，故明确区分观测标记与连续性推断；原指令从桌面移至Desktop/Claude命令后按同名文件定位并保存hash；首次仅因旧指令路径不可用而在任何统计输出前退出，没有重训。
只读PDF提取曾遇控制台编码问题，改用UTF-8；绘图前两版存在标题/坐标标签覆盖，已保留版本并在第三版显式关闭自动布局后通过程序与彩色/灰度目视检查。旧实验产物未覆盖。
本阶段没有A/B标签效应判定、没有BC上限H判定、没有近/远选择判定；这些均须后续授权和数据。未改变V9任何决策规则。

代码目录：`C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\oracle_diag\v9_hopper_label_headroom`。输出目录：`C:\Users\TXin\Documents\New project\off-dynamics-rl-main\off-dynamics-rl-main\outputs\v9_hopper_label_headroom_20261030`。
配置：config.json、PLOT_CONFIG_v3.json；数据/版本保护：INPUT_MANIFEST.json、OLD_FILE_PROTECTION_BEFORE.json；最终凭据：STAGE0_COMPLETED.json。

实际命令（工作目录为项目根目录；所有Python均-B，CUDA_VISIBLE_DEVICES=-1、PYTHONDONTWRITEBYTECODE=1）：

```powershell
& 'D:/ANACONDA/envs/offdynamics/python.exe' -B 'oracle_diag/v9_hopper_label_headroom/stage0_diagnose.py'
& 'C:/Users/TXin/.codex/environments/scientific-figures/.venv/Scripts/python.exe' -B 'oracle_diag/v9_hopper_label_headroom/stage0_plot.py'
& 'C:/Users/TXin/.codex/environments/scientific-figures/.venv/Scripts/python.exe' -B 'oracle_diag/v9_hopper_label_headroom/stage0_plot.py' --vector-after-qa
& 'D:/ANACONDA/envs/offdynamics/python.exe' -B 'oracle_diag/v9_hopper_label_headroom/stage0_finalize.py'
```

PDF只读渲染使用bundled Poppler pdftoppm（第22、23页）；图的程序合规检查使用scipilot-figure-skill/scripts/check_figure.py --strict，最终PNG与SVG均PASS。

**按用户本轮授权，阶段0到此停止，等待阶段1的明确确认。** 阶段1若获批，仅生成Hopper正常源数据在morph动力学下的全量oracle标签，并独立回放至少10000条验收，不包括任何训练。
阶段1暂预留约0.5–2小时CPU作为规划量，并非Hopper实测承诺：历史Walker全量生成记录675.05秒（约11.25分钟），Hopper隐藏状态恢复、失败诊断及独立验收耗时尚未知；验收不通过则停止，不能进入阶段2。阶段2/3/4均未启动。
