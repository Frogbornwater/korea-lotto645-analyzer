# 韩国 Lotto 6/45 V3：官方历史数据驱动分析实验室

V3 的重点是官方历史数据、候选号码策略比较、滚动样本外回测。它不承诺也不声称能准确预测随机彩票开奖结果。

## 文件结构

```text
app.py
lotto_core.py
sync_data.py
requirements.txt
README.md
.github/workflows/refresh-data.yml
data/                 # 工作流成功后生成 CSV 和同步状态
```

## 通过 GitHub 网页升级（Mac 不需要安装 Python）

仓库：<https://github.com/Frogbornwater/korea-lotto645-analyzer>

1. 下载并解压本项目 ZIP。
2. 在仓库根目录上传并替换 `app.py`、`lotto_core.py`、`sync_data.py`、`requirements.txt`、`README.md`。
3. 在 GitHub 中创建 `.github/workflows/refresh-data.yml`，把本包同路径文件的完整内容粘贴进去。GitHub 编辑器可自动创建 `.github/workflows` 目录。
4. 在仓库 Settings → Actions → General → Workflow permissions，选择 `Read and write permissions` 并保存。
5. 进入 Actions → “同步韩国官方 Lotto 历史数据” → Run workflow。首次任务会按官方站点提供的查询窗口抓取第 1 期至最新期，并检查期号是否连续。
6. 成功后应在仓库看到 `data/lotto645.csv` 与 `data/sync_status.json`。Streamlit Community Cloud 通常会检测提交并重新部署。
7. 如果 Actions 失败，先看失败日志。脚本不会用第三方镜像悄悄补齐，也不会覆盖上一份 CSV。

## 数据来源

- 官方开奖查询页面：<https://www.dhlottery.co.kr/lt645/result>
- 官方网站前端使用的最新期数据路径：`https://www.dhlottery.co.kr/selectMainInfo.do`
- 官方网站前端使用的指定期次查询路径：`https://www.dhlottery.co.kr/lt645/selectPstLt645InfoNew.do?srchDir=center&srchLtEpsd={回次}`
- 关键字段：`ltEpsd`（期号）、`ltRflYmd`（开奖日期）、`tm1WnNo` 至 `tm6WnNo`（六个号码）、`bnsWnNo`（奖金号）。

重要：这些是官方站点页面使用的 JSON 查询路径，不是保证长期稳定的版本化公共 API。站点改版、维护或访问限制可能导致同步失败。同步失败时应保留已有数据并检查 GitHub Actions 日志。

首次同步会尝试完整抓取官方历史期数，使用重叠窗口查询并对缺失期号再次请求；仅当从第 1 期到最新期完整连续时，才会写入 CSV。V3 不以第三方镜像作为初始官方数据替代。

## V3 模型

- 近期频率模型：指定近期窗口内的号码出现频率，并向理论频率收缩。
- 长期频率模型：使用全部已收录历史开奖频率。
- 指数时间衰减模型：按距今期数对历史记录加权。
- 贝叶斯平滑模型：使用 Beta-Binomial 风格的先验平滑，降低小样本极端值。
- 综合评分模型：短期、中期、长期统计加权组合。
- 历史结构约束模型：按历史和值分位数和宽松结构约束筛选组合。
- 均匀随机基准：作为回测对照。

“模型相对得分”是当前样本下的相对排序分，不是任何号码的中奖概率。遗漏期数只作为历史描述，不假设遗漏越久越容易开奖。

## 回测

采用 walk-forward：对每一个测试期，模型只使用此前开奖数据生成候选组合。报告平均命中数、3+ 命中率等指标，并以按测试期重采样的 Bootstrap 区间比较随机基准。历史领先不保证未来领先；对每注固定六号码，理论头奖概率仍为 `1 / C(45,6) = 1 / 8,145,060`。

## 本地测试（可选）

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Streamlit Cloud 会自动安装 requirements.txt；用户的 Mac 不需要安装 Python。
