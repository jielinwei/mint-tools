# MINT Tools 中文说明

这是一个用于放疗 DICOM 数据审计的 Python 工具集，重点解决以下问题：

- 识别 CT / RTDOSE / RTPLAN / RTSTRUCT / MR
- 建立 RT 对象之间的引用关系
- 识别候选 planning CT
- 输出适合人工复核的 CSV

## 适用范围

- 只读审计原始数据
- 不修改原始 DICOM
- 不进行分割、重采样、radiomics

## 主要脚本

- `scripts/audit_mint_rt_dicom.py`
  基础 RT DICOM 审计
- `scripts/analyze_mint_folder_patterns.py`
  planning CT 文件夹模式分析
- `scripts/rank_planning_ct_candidates.py`
  planning CT 候选排序
- `scripts/export_planning_ct_nifti.py`
  基于人工确认后的 resolved CSV 导出 planning CT NIfTI

## 使用原则

- 原始数据目录只读
- 输出写入本地 `outputs/` 和 `logs/`
- 建议先 `--dry-run`，再正式运行
- 如需按人工整理的文件夹知识库筛选，可使用 `--knowledge-xlsx` 和 `--knowledge-sheet`

## Windows 运行

1. 创建虚拟环境
2. 安装 `requirements.txt`
3. 复制 `config.example.yaml` 为 `config.yaml`
4. 修改 `source_dir` 和 `output_dir`
5. 运行：

```powershell
.\run_audit_windows.ps1 -SourceDir "D:\raw_data" -OutputDir "."
```

## 按 Excel 知识库筛选

```powershell
python scripts\audit_mint_rt_dicom.py --source-dir "D:\raw_data" --output-dir . --dry-run --knowledge-xlsx "D:\mint-tools\folder_structure3_categeried2026_RT_RT232_matched2.xlsx" --knowledge-sheet "RT243-3012-final"
python scripts\analyze_mint_folder_patterns.py --source-dir "D:\raw_data" --output-dir outputs --dry-run --knowledge-xlsx "D:\mint-tools\folder_structure3_categeried2026_RT_RT232_matched2.xlsx" --knowledge-sheet "RT243-3012-final"
```

筛选规则：只保留原始数据路径中 `<ID>/<Date>/...` 与 Excel 指定 sheet 中 `ID`、`Date` 组合匹配的文件。这里的 `Date` 是第二层检查文件夹完整名称，例如 `20130710-RTLUNG_MAMMA-CUR-BST`。

## 导出 planning CT NIfTI

人工确认 `mint_manual_review_final_planning_ct_resolved.csv` 后再运行：

```powershell
python scripts\export_planning_ct_nifti.py --source-dir "D:\raw_data" --resolved-csv "outputs\mint_manual_review_final_planning_ct_resolved.csv" --output-dir "outputs\planningCT" --dry-run
python scripts\export_planning_ct_nifti.py --source-dir "D:\raw_data" --resolved-csv "outputs\mint_manual_review_final_planning_ct_resolved.csv" --output-dir "outputs\planningCT"
```

输出命名格式：`patientID_YYYYMMDD_planningCT.nii.gz`。如果 `final_ct_series_uid` 存在，脚本会在整个 patient 文件夹下聚合同一个 `SeriesInstanceUID` 的 CT DICOM，避免 MINT 中同一 CT series 被拆到多个文件夹时漏层。

## 注意

- `outputs/`、`logs/`、人工审核结果等数据产物不建议提交到 GitHub
- GitHub 仓库中应只保留脚本、示例配置和说明文件
