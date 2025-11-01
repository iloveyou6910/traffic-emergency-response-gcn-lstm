# 代码模板包使用说明

## 目录
- `config.template.yaml`：配置模板（可复制为 `config.yaml` 并修改）。

## 快速使用
1. 复制模板：
   - 将 `templates/config.template.yaml` 复制为 `my_project/config.yaml`
2. 安装依赖：
   - `pip install -r my_project/requirements.txt`
3. 运行演示：
   - Windows（PowerShell）：`./my_project/run_demo.ps1`

## 可选扩展
- 使用外部数据集：按 `docs/dataset_links_and_templates.md` 的“开源数据集链接”获取数据，并在 `config.yaml` 中调整 `paths.data_dir`。
- 调参与复现：通过前端侧边栏保存/加载配置、训练与路由参数持久化到 `config.yaml`。