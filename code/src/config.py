# 配置参数 - 激进优化版
# 任务: 筛选5只股票, 分配权重购入, 持有5天不操作
sequence_length = 60
feature_num = '158+39'
config = {
    # ---- 数据参数 ----
    'sequence_length': sequence_length,
    'feature_num': feature_num,

    # ---- 模型超参数 ----
    'd_model': 256,
    'nhead': 4,
    'num_layers': 3,
    'dim_feedforward': 512,
    'dropout': 0.1,

    # ---- 特征注意力参数 (激进) ----
    'feature_attention_temperature': 0.5,  # 降低温度使注意力更集中

    # ---- 训练参数 ----
    'batch_size': 4,
    'num_epochs': 50,
    'learning_rate': 1e-5,
    'max_grad_norm': 5.0,
    'gradient_accumulation_steps': 1,  # 梯度累积步数 (>1 模拟大batch)

    # ---- 权重衰减 ----
    'weight_decay': 1e-5,

    # ---- 学习率调度器 (激进) ----
    'lr_scheduler': 'cosine_restart',  # 'cosine_restart' | 'linear' | 'onecycle'
    'cosine_T_0': 10,      # 热重启周期
    'cosine_T_mult': 2,    # 周期倍增因子
    'cosine_eta_min': 1e-6,

    # ---- 直接优化组合收益率 ----
    'direct_return_optimization': True,  # True=直接优化组合收益率, False=优化排序
    'return_weight': 1.0,     # 收益率损失权重
    'risk_penalty': 0.1,      # 风险惩罚项权重 (激进: 较小)

    # ---- 排序损失参数 (仅当 direct_return_optimization=False) ----
    'loss_temperature': 0.5,  # 损失温度参数 (激进: 降低温度)
    'pairwise_weight': 2.0,   # 配对损失权重 (激进: 提高到2.0)
    'base_weight': 1.0,
    'top5_weight': 5.0,       # top-5样本权重 (激进: 提高到5.0)

    # ---- 权重分配参数 ----
    'max_single_weight': 0.3,    # 单只股票最大权重 (激进: 允许50%)
    'weight_temperature': 3.0,   # 权重分配温度 (越大分配越集中)

    # ---- 路径 ----
    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data/data-03',
}
