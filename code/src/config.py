# 配置参数
sequence_length = 60
feature_num = 'dragon'  # 可选: '39', '158+39', 'strategy', 'strategy+base', 'dragon'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据
    'model_type': 'transformer',         # 'transformer' 或 'linear'
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些
    'num_epochs': 50,       # 排序任务可能需要更多epochs
    'learning_rate': 1e-5,  # 学习率
    'dropout': 0.3,         # 增加dropout防止过拟合
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1,   # 配对损失权重
    'base_weight': 1.0,     # 非top-k样本权重
    'top5_weight': 2.0,     # top-5样本额外权重

    # 策略优化参数
    'use_strategy_features': False,   # 龙头模式不需要旧策略特征
    'strategy_sample_weight': 2.0,     # 符合策略条件的样本额外权重
    'strategy_score_threshold': 4,     # str_total_score >= 此值视为符合策略
    'loss_function': 'lambdarank',    # 'lambdarank' | 'weighted_ranking'
    'label_type': 'rank',  # 'future5d_return' | 'rank'
    'feature_selection_keep': 100,    # IC过滤后保留特征数，0=不过滤

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}
