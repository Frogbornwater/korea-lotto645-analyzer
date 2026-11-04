# 韩国 Lotto 6/45 V3.1：数据同步与模型逻辑修正版

V3.1 是对原 V3 仓库审查后的修正版。它改进官方接口兼容、失败状态可见性、重复数据处理、候选生成和回测覆盖范围。系统只对历史样本作统计研究，不声称能预测随机开奖或提高单注中奖概率。

## 文件结构

```text
app.py
lotto_core.py
sync_data.py
requirements.txt
README.md
.github/workflows/refresh-data.yml
tests/test_lotto_core.py
data/                 # 首次成功同步后生成
```

## 在 GitHub 网页更新（Mac 不需要安装 Python）

1. 将本 ZIP 解压。
2. 在仓库根目录上传并替换 `app.py`、`lotto_core.py`、`sync_data.py`、`requirements.txt`、`README.md`。
3. 创建或替换 `.github/workflows/refresh-data.yml`。
4. 新增 `tests/test_lotto_core.py`（在仓库根目录创建 `tests` 文件夹）。
5. 确认仓库 Settings → Actions → General → Workflow permissions 为 `Read and write permissions`。
6. 在 Actions 中运行 `Refresh Lotto 6/45 data`。
7. 同步成功后，仓库会生成 `data/lotto645.csv` 和 `data/sync_status.json`，Streamlit Cloud 会因新提交重新部署。

## V3.1 修正点

- 官方开奖查询尝试新版 `selectPstLt645InfoNew.do`，并回退到同站点 `selectPstLt645Info.do`。
- 解析器递归识别官方 JSON 嵌套层级，不再只依赖固定的一种 JSON 外壳。
- 官方响应重复期次如存在冲突，停止同步，不再静默覆盖。
- CSV 使用临时文件写入后原子替换，完整校验通过后才覆盖旧数据。
- GitHub Actions 即使同步步骤失败，也会提交 `sync_status.json`，使失败原因能在应用页面读取；历史 CSV 不会因失败而覆盖。
- 候选组数与重号限制冲突时，逐步放宽重号上限，但继续使用所选模型；界面显示请求上限、实际允许上限和实际最大重号。
- “模型相对得分”改为基于当前所选模型的号码权重计算；随机和结构约束模型显示中性分数，不再错误使用五模型平均分。
- 将“历史结构约束模型”加入滚动样本外回测。
- 新增 8 项可复现单元测试。

## 官方数据来源说明

数据仅从 동행복권 官方网站页面使用的 JSON 查询路径获取：

- 官方结果页：<https://www.dhlottery.co.kr/lt645/result>
- 最新页面数据：`https://www.dhlottery.co.kr/selectMainInfo.do`
- 历史期次查询首选：`https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do`
- 备用官方历史路径：`https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do`

这些是官网前端使用的查询路径，并不是具有长期稳定性保证的公开 API。网站维护、限流或反自动化限制仍可能导致同步失败。失败时会保留此前有效 CSV，并更新失败状态。

**当前这个 ZIP 不含 `data/lotto645.csv`**：历史记录需要你在 GitHub Actions 中首次运行官方同步后由程序生成。没有 CSV 时，Streamlit 页面会明确提示先同步，不会生成伪造的历史分析结果。

## 模型与概率边界

- 近期频率、长期频率、指数时间衰减、贝叶斯平滑、综合评分：按历史频率为号码设置相对抽样权重。
- 历史结构约束：根据历史和值宽分位和宽松奇偶/区间约束筛选组合。
- 均匀随机基准：对照组。
- 回测采用 walk-forward，每个测试期只使用此前开奖记录；随机种子、模型、测试窗口应在比较时保持一致。
- 模型相对得分不是中奖概率。公平独立开奖时，任何一组固定的六号码头奖概率均为 `1 / C(45,6) = 1 / 8,145,060`。

## 本地测试（可选）

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Streamlit Cloud 会自动安装 `requirements.txt`；使用在线应用的 Mac 不需要安装 Python。
