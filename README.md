# Lotto 6/45 V3.6.1 — 综合升级版（离线手动数据）

V3.6.1 在 V3.6 第一阶段的基础上，一次性加入参数搜索、多模型集成、10组组合优化和模型实验档案。

## 固定数据政策

- **禁止连接韩国 Lotto / 동행복권 官方网站及其 API。**
- 禁止自动爬取、定时同步或通过后台任务获取开奖数据。
- 仅分析用户手动导入的 CSV、XLSX、XLS，或仓库 `data/lotto645.csv` 中由用户手动维护的数据。
- 不要把回测明细 CSV 当成开奖历史数据。

## 文件结构

```text
app.py
lotto_core.py
backtest_engine.py
parameter_optimizer.py
portfolio_optimizer.py
model_registry.py
model_management.py
data_manager.py
sync_data.py                 # 安全停用桩，不访问网络
requirements.txt
README.md
data/                         # 保留用户原有 data/lotto645.csv
baseline/                      # 历史回测基准资料，不是开奖数据库
tests/
```

## V3.6.1 功能

1. **独立回测引擎**：逐期 walk-forward；每个目标期只使用此前数据；保存每注号码、训练截止期号、模型版本和实验参数。
2. **自动参数搜索**：近期频率窗口、贝叶斯先验、指数衰减半衰期、综合评分权重、多模型集成权重。搜索排名为探索性结果，不计算或宣称参数搜索冠军具有显著优势。
3. **保留区间验证**：参数搜索前封存最新一段历史；单独用固定参数评估。查看结果后该区间即已使用，不得反复调参并继续称其为未触碰测试集。
4. **多模型集成**：新增“多模型集成模型”，融合近期频率、贝叶斯平滑、指数时间衰减。默认权重为 0.40 / 0.35 / 0.25，可在页面调整。
5. **10组组合优化**：从模型加权候选池中，以贪心启发式平衡号码相对评分、覆盖增量和组间重叠。报告覆盖率、最大/平均重号和实际放宽情况。
6. **模型管理**：维护模型状态和实验摘要；支持 JSON 导入/导出。Streamlit Cloud 会话状态不是长期数据库，建议每次重要实验后下载档案。

## 参数搜索默认网格

- 近期频率窗口：30、50、80、100、150、200、300 期
- 贝叶斯先验强度：20、40、60、80、120、160、240
- 指数衰减半衰期：20、30、40、60、80、120、160 期
- 综合评分：多组短/中/长期权重组合
- 多模型集成：多组近期/贝叶斯/衰减权重组合

参数网格可在 `parameter_optimizer.py` 的 `default_parameter_grid()` 中调整。新增网格必须继续遵守时间顺序验证。

## 部署（GitHub 网页，无需本地安装 Python）

1. 先下载当前 GitHub 仓库 ZIP 备份。
2. 在 GitHub Actions 中禁用旧的自动同步工作流，并从仓库删除会触发外部开奖数据读取的旧 workflow YAML。不要保留定时爬取任务。
3. 解压本升级包。
4. 将本包中的下列文件放入仓库根目录并覆盖同名文件：
   - `app.py`
   - `lotto_core.py`
   - `backtest_engine.py`
   - `model_registry.py`
   - `data_manager.py`
   - `sync_data.py`
   - `requirements.txt`
   - `README.md`
5. 新增以下文件到仓库根目录：
   - `parameter_optimizer.py`
   - `portfolio_optimizer.py`
   - `model_management.py`
6. 上传 `tests/` 文件夹（推荐，用于维护和后续版本回归测试）。
7. **保留仓库现有的 `data/lotto645.csv` 和 `data/sync_status.json`（如有）；不要用 baseline 文件替换。**
8. 不要上传旧 `.github/workflows` 中已停用的自动同步工作流；确认仓库没有其他自动拉取开奖数据的 workflow。
9. Commit changes，等待 Streamlit Cloud 自动重新部署。

如果旧仓库存在 `data/` 文件夹，上传时只替换指定根目录代码文件；不要上传本包的 `baseline/` 去覆盖或移动原始开奖数据。`baseline/` 可以作为独立档案目录保留，也可不上传。

## 建议首次验收

- 手动导入历史 CSV / Excel 正常。
- 七种原有策略 + 多模型集成模型，共八种策略可以进行标准回测。
- 100 期 × 3 组种子 × 10 注 × 8 策略 = 24,000 条逐注明细。
- 参数搜索只读取开发/验证段，不读取保留尾段。
- 固定参数保留区间验证能与随机基准配对比较。
- 组合优化每次生成 10 组互不相同且合法的号码。
- 模型档案 JSON 可以下载、导入和恢复。
- 应用源码没有 HTTP/API/网络数据抓取逻辑。

## 自动测试与部署验收

- 自动测试报告：`V36_PHASE1_TEST_REPORT_CN.md` 和 `V36_PHASE2_4_TEST_REPORT_CN.md`。
- 本地已通过 31 项 unittest 和 Python 编译检查；Streamlit Cloud 页面仍需用户部署后端到端验收。

## 统计解释边界

公平独立开奖下，每一注固定的六号码组合头奖概率相同，为 1 / 8,145,060。历史频率、遗漏、共现、模型评分和组合优化不改变单注理论中奖概率。参数排名、历史回测领先或组合覆盖率较高，都不等于存在未来预测优势。
