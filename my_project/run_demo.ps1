# 一键演示脚本
# 功能：自动执行数据生成→训练→预测→路径规划→解释报告

# 1. 生成数据
Write-Host "生成合成数据..."
python src/prepare_data.py --config config.yaml

# 2. 训练模型
Write-Host "开始训练..."
python src/train_eval.py --config config.yaml

# 3. 运行预测
Write-Host "生成预测结果..."
python src/inference.py --config config.yaml

# 4. 路径规划与解释报告
Write-Host "启动前端交互界面..."
streamlit run src/app_streamlit.py

Write-Host "演示完成！访问 http://localhost:8501 查看结果。"