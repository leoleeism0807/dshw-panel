# 上市公司资本结构影响因素分析

> [作业要求](https://github.com/lianxhcn/dsfin/blob/main/homework/ex_P03_Panel-capital_strucuture.md)

### GitHub 仓库链接

待创建远程仓库 `dshw--panel` 并发布后补充。

### Quarto Book 链接

待 GitHub Pages 发布后补充。

## 个人信息

- 姓名：李泽欣
- 学号：25210067
- 邮箱：15532312611@163.com

## 数据来源

- CSMAR，下载时间：2026-04-20
- 最终样本：4,775 个公司，40,846 个观测值，2010-2025 年

## 项目结构

- `notebooks/01_data_construction.ipynb`
  原始数据整理、变量构造与作业 1.3 样本筛选
- `notebooks/02_empirical_analysis.ipynb`
  描述性统计、图形、M1-M4 回归结果与文字分析
- `data/raw/`
  原始 CSMAR 下载文件
- `data/clean/`
  清洗后的年度面板数据与筛选后样本
- `output/figures/`
  作业图形输出
- `output/tables/`
  作业表格输出
- `scripts/`
  可复现 Python 脚本
- `chapters/`、`index.qmd`、`_quarto.yml`
  Quarto Book 章节稿与配置文件
- `docs/`
  Quarto 渲染后的静态网页输出目录，可用于 GitHub Pages

## 样本筛选流程

| 筛选步骤 | 剔除观测数 | 剩余观测数 | 剩余公司数 |
|---|---:|---:|---:|
| 初始样本 | — | 57,605 | 5,671 |
| 剔除金融保险 | 1,477 | 56,128 | 5,616 |
| 剔除 ST/PT | 10,857 | 45,271 | 4,888 |
| 剔除 Lev > 1 | 20 | 45,251 | 4,888 |
| 剔除缺失值 | 4,405 | 40,846 | 4,775 |
| 最终样本 | — | 40,846 | 4,775 |

## 工具

- Python 3
- Jupyter Notebook
- pandas / statsmodels / linearmodels / pyfixest（环境辅助）

## 变量构造说明

- `Lev = 总负债 / 总资产`
- `NPR = 净利润 / 总资产`
- `Size = ln(总资产)`
- `Tang = 固定资产净额 / 总资产`
- `Growth = (总资产_t - 总资产_{t-1}) / 总资产_{t-1}`
- `NDTS = (固定资产折旧 + 无形资产摊销 + 长期待摊费用摊销) / 总资产`
- `SOE = 1` 表示国有企业，`SOE = 0` 表示非国有企业

说明：

- 财务报表使用合并报表年末值。
- 为严格按照题意计算 2010 年成长率，数据构造阶段保留了 2009 年年末总资产作为滞后缓冲期。
- `Lev`、`NPR`、`Tang`、`Growth`、`NDTS` 在年度截面内均执行双侧 1% Winsorize。

## 主要发现

1. 双向固定效应模型显示，盈利能力 `NPR` 对杠杆率 `Lev` 的估计系数为 `-0.625`，且在 1% 水平上显著。这意味着在控制公司固定效应、年度固定效应及核心控制变量后，盈利能力更高的企业反而配置了更低的杠杆率，整体结论支持优序融资理论。
2. 在交互固定效应稳健性检验中，`NPR` 的系数仍为显著负值，估计值约为 `-0.405`；同时，`M2` 增长率系数显著为正，说明货币扩张环境会推动总体杠杆率上升，但不会改变 `NPR-Lev` 关系的基本方向。
3. 分组回归结果显示，国有企业样本中 `NPR` 的负向系数约为 `-0.858`，非国有企业约为 `-0.510`。交互项模型进一步表明，`NPR × SOE` 系数显著为负，说明产权性质会显著强化盈利能力对杠杆率的负向影响。
4. 时变系数模型表明，2010-2025 年间 `NPR` 对 `Lev` 的影响始终为负，但系数绝对值存在明显阶段性波动。2018 年前后负向关系相对减弱，而 2023-2025 年的负向效应显著增强，说明企业融资行为在不同宏观阶段呈现出明显异质性。

## Quarto Book

- Quarto 书稿入口文件：`index.qmd`
- 章节目录：`chapters/`
- Quarto 配置：`_quarto.yml`
- 本地渲染输出：`docs/`
- 本地预览入口：`docs/index.html`

章节结构如下：

1. `第一章 研究背景与研究假设`
2. `第二章 数据来源、变量构造与样本筛选`
3. `第三章 实证结果`
4. `第四章 稳健性检验`
5. `第五章 结论`

本书稿基于已经生成的 `output/tables/` 与 `output/figures/` 组织内容，能够作为课程作业的网页展示版本，与 Notebook 中的分析结果保持一致。

## GitHub 仓库与发布说明

- 建议 GitHub 仓库名称：`dshw--panel`
- 仓库应设置为 `Public`
- 根据 `.gitignore`，原始数据目录 `data/raw/` 不会被纳入版本控制
- 可纳入仓库的内容包括：
  - `data/clean/`
  - `output/`
  - `notebooks/`
  - `chapters/`
  - `docs/`
  - `scripts/`

GitHub Pages 发布完成后，README 中建议填写链接格式如下：

- `https://<your-github-username>.github.io/dshw--panel/`

说明：

- 当前本地目录已完成 Quarto Book 配置和渲染输出准备；
- 当前项目已经初始化为本地 git 仓库，默认分支为 `main`；
- 若需要真正发布到 GitHub Pages，仍需在你自己的 GitHub 账户下创建 `dshw--panel` 仓库、推送代码，并在仓库设置中启用 Pages。
- 推荐的 Pages 设置为：
  - Source：`Deploy from a branch`
  - Branch：`main`
  - Folder：`/docs`

## 文件说明

- `notebooks/01_data_construction.ipynb`
  负责原始表结构检查、变量构造、年度面板合并以及作业第 1.3 节样本筛选。
- `notebooks/02_empirical_analysis.ipynb`
  负责从 clean 数据出发，重新执行样本筛选、异常值处理、描述性统计、图形绘制、回归估计和文字分析。Notebook 中包含生成全部表格和图形的完整代码，点击运行即可复现结果。
- `data/clean/panel_capital_structure_annual.csv`
  年度总表，保留作业所需变量。
- `data/clean/panel_capital_structure_filtered.csv`
  按作业第 1.3 节筛选后的最终分析样本。
- `output/tables/Table1-Table5`
  对应作业主要表格输出。
- `output/figures/Fig1-Fig5`
  对应作业主要图形输出。
- `docs/`
  Quarto Book 的静态网页成品目录，可直接用于 GitHub Pages。


