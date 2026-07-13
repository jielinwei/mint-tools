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

## 使用原则

- 原始数据目录只读
- 输出写入本地 `outputs/` 和 `logs/`
- 建议先 `--dry-run`，再正式运行

## Windows 运行

1. 创建虚拟环境
2. 安装 `requirements.txt`
3. 复制 `config.example.yaml` 为 `config.yaml`
4. 修改 `source_dir` 和 `output_dir`
5. 运行：

```powershell
.\run_audit_windows.ps1 -SourceDir "D:\raw_data" -OutputDir "."
```

## 注意

- `outputs/`、`logs/`、人工审核结果等数据产物不建议提交到 GitHub
- GitHub 仓库中应只保留脚本、示例配置和说明文件
