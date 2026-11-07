# 结果证据的正确口径

- runtime/artifacts/training.json：所选最终权重的原始元数据，固定截距b=−0.5796672067792444。里面附带confirmation字段记录的是最终重拟合之前的确认阶段。
- selected_version_evidence.json：当前所选版本的核查汇总，含模型参数、运行包摘要和单次线上六预算曲线。
- base_training_metadata.json：原最终基础配方训练记录与数据审计，150000行、105个公开二分类任务。
- training_benchmarks_105.txt：公开训练任务名称。
- selection_historical.json：V10校准规格选择记录；报告方法级交叉拟合结果时可引用，不是固定最终b的独立测试。
- confirmation_historical.json：规格冻结后、最终758episode重拟合之前的确认结果。该阶段的b不同于最终b；不要把其分数归给最终固定参数。
- prepared.json：V10验证基础训练123060行、开发698episode、后续确认60episode的记录；不是最终150000行基础训练元数据。
- calibration_refit_check.json：本次独立入口重新拟合最终b通过。只核实拟合参数，不计算新的隐藏测试成绩。
- package_check.json：历史原运行包隔离检查通过及计时，不代表模型精度。
- development_runtime_check.json、confirmation_runtime_check.json：历史快速评分器与真实接口对齐，针对当时相应参数阶段。
- runtime_source_parity.json：原ZIP推理模块与本次收集的对应源代码逐字节一致。
- selected_budget_curve.csv：1020550单次线上结果均值；ALC复算0.13951353333333336，平台显示舍入为0.139514。
- same_version_submissions.csv：用户提供的9次同名V10已完成记录，描述性均分约0.184069，范围0.139514–0.218715。样本不是完全配对，不用于声称稳定期望或严格置信区间。

calibration_cache中的标签来自公开任务，不是网站隐藏评测反馈。最终校准重拟合使用原五个开发任务及后续公开确认任务，所以这些参与拟合的数据不再是最终截距的独立测试。

本次新增工具的打包检查结果见handoff_validation.json（若存在）。没有重新执行基础模型150000行训练或完整历史候选搜索。
