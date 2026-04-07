# 导出 slides（PPT/PDF）

源文件：`docs/teacher_report_20260210_slides.md`

## 方式 1：Marp CLI（推荐）

```bash
npm install -g @marp-team/marp-cli
marp docs/teacher_report_20260210_slides.md --pptx -o docs/teacher_report_20260210_slides.pptx
marp docs/teacher_report_20260210_slides.md --pdf  -o docs/teacher_report_20260210_slides.pdf
```

## 方式 2：不装全局，直接 npx

```bash
npx @marp-team/marp-cli docs/teacher_report_20260210_slides.md --pptx -o docs/teacher_report_20260210_slides.pptx
```

## 方式 3：VS Code 插件

1. 安装插件：`Marp for VS Code`
2. 打开 `docs/teacher_report_20260210_slides.md`
3. 右上角 `Open Preview`
4. `Export Slide Deck` 选择 `PPTX` 或 `PDF`
