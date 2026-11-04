# 韩国 Lotto 6/45 智能分析台（云端部署版）

这是用于 Streamlit Community Cloud 的部署包。部署后通过浏览器访问，无需在个人电脑安装 Python。

## 上传到 GitHub 的文件

将本目录内的三个文件直接上传到 GitHub 仓库根目录：

- `app.py`
- `requirements.txt`
- `README.md`

不要把整个 ZIP 文件上传后直接部署；先在 Mac 上双击 ZIP 解压，再上传解压后目录中的文件。

## Streamlit Community Cloud 部署

1. 创建或登录 GitHub 账号：<https://github.com/>。
2. 新建一个仓库，建议命名为 `korea-lotto645-analyzer`。初次部署建议选择 Public，且不要勾选自动创建 README（本包已包含 README）。
3. 进入仓库，选择 **Add file → Upload files**。
4. 将上面列出的四个文件拖入上传框，提交（Commit changes）到 `main` 分支。
5. 打开 <https://share.streamlit.io/>，使用 GitHub 登录并授权 Streamlit 访问该仓库。
6. 选择 **Create app → Yup, I have an app**。
7. Repository 选择刚才创建的仓库；Branch 选 `main`；Main file path 填 `app.py`。
8. 可选地设置一个未被占用的 App URL，然后点击 **Deploy**。

部署完成后，页面会显示 `https://....streamlit.app` 网址。保存该网址即可在 Mac、手机或其他电脑的浏览器中打开。

## 本版数据说明

- 首次打开时会尝试从公开历史 JSON 镜像读取开奖数据。
- 镜像并非官方发布渠道；请通过页面中的韩国 동행복권 官方结果链接核对最新开奖。
- 若数据镜像无法访问，可通过左侧栏导入符合页面说明的 CSV。
- Community Cloud 容器的本地文件不应视为永久存储；重启或重新部署后，缓存文件可能需要重新获取或重新导入。
- 不需要配置 API Key 或 Secrets。

## 免责声明

韩国 Lotto 6/45 是随机抽奖。历史频率、遗漏值、共现统计和回测不能可靠预测下一期开奖号码，也不能提高单注理论头奖概率。本工具用于统计研究与娱乐，不构成投注建议。
