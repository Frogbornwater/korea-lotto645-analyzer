# 韩国 Lotto 6/45 AI 研究台 V2

V2 在原 Streamlit 应用上增加了 2026 年 동행복권 结果页 JSON 字段兼容、数据完整性检查、多策略候选实验、带多随机种子和按期 Bootstrap 区间的滚动样本外回测，以及 GitHub Actions 定时更新历史 CSV。

## 本次项目结构

```text
app.py
lotto_core.py
requirements.txt
README.md
sync_data.py
.github/workflows/refresh-data.yml
tests/test_lotto_core.py
data/                 # 自动同步后保存历史 CSV 和同步状态
```

## 在 GitHub 网页更新（不需要在 Mac 安装 Python）

1. 解压 V2 ZIP。
2. 打开你的仓库：<https://github.com/Frogbornwater/korea-lotto645-analyzer>。
3. 将根目录的 `app.py`、`lotto_core.py`、`sync_data.py`、`requirements.txt`、`README.md` 上传到仓库根目录；勾选替换现有 `app.py`、`requirements.txt`、`README.md`。
4. 在仓库首页点击 **Add file → Create new file**，文件名填写 `.github/workflows/refresh-data.yml`。打开 ZIP 解压目录里的根目录文件 `refresh-data.yml`，复制全部内容，粘贴到 GitHub 编辑框，然后 Commit changes。GitHub 会自动创建 `.github/workflows` 两层目录。
5. 点击 Commit changes。Streamlit Community Cloud 会检测仓库更新并重新部署；第一次部署后请确认 App 页面正常。
6. 到 GitHub 仓库的 **Actions** 标签页，打开 **Refresh Lotto 6/45 data**，点击 **Run workflow** 手动执行首次同步。成功后仓库会出现 `data/lotto645.csv` 与 `data/sync_status.json`。
7. 以后工作流计划在韩国时间周六、周日约 23:00 检查开奖数据；GitHub 计划任务可能延迟，必要时可在 Actions 手动运行。
8. 可选：上传 `tests/test_lotto_core.py` 与 `tests/conftest.py`，后续可加自动化测试。

## 数据来源和边界

- 优先读取 동행복권结果页面使用的公开 JSON 查询路径：`/lt645/selectPstLt645Info.do` 和 `/lt645/selectPstLt645InfoNew.do`。
- 这些是官网页面使用的接口路径，并非官方承诺长期稳定的开放 API。网站改版后可能变化。
- 初始完整历史数据使用公开镜像作为引导；如镜像不可用，脚本尝试通过官网结果页 JSON 分段补全。
- 当多个数据源存在重叠期次时，官方结果页数据优先；数据缺期会显示警告。
- GitHub Actions 把历史 CSV 提交回仓库，避免仅依赖 Streamlit Cloud 容器临时磁盘。
- 工作流首次运行需要 GitHub Actions 有仓库内容写入权限：Repository Settings → Actions → General → Workflow permissions → Read and write permissions。

## V2 统计方法

- 号码频率显示理论期望、标准化偏差 Z 与未经多重比较校正的近似 P 值。
- 两数组合共现以每个指定号码对单期开奖理论概率 1/66 为参考。
- 候选策略包括均匀随机、近期频率倾向、指数时间衰减、遗漏倾向和结构约束随机。
- 滚动回测采用仅使用过去数据的 walk-forward 设计，随机种子重复，并以测试期为单位 Bootstrap 估计策略与随机基准的差值区间。

## 运行依赖

Streamlit Cloud 会根据 `requirements.txt` 自动安装依赖。个人电脑仍然不需要安装 Python。

## 免责声明

Lotto 6/45 是随机抽奖。历史统计、遗漏、共现、机器学习和回测不能可靠预测下一期开奖号码，也不会提高单注理论头奖概率。本项目用于数据研究和娱乐，不构成投注建议。
