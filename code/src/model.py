import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EnhancedCrossStockAttention(nn.Module):
    """增强版股票间交互注意力模块 - 添加前馈网络提升表达能力"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(EnhancedCrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_features):
        # stock_features: [batch, num_stocks, d_model]
        attended, attn_weights = self.cross_attention(stock_features, stock_features, stock_features)
        out1 = self.norm1(stock_features + self.dropout(attended))
        ff_out = self.feed_forward(out1)
        out2 = self.norm2(out1 + self.dropout(ff_out))
        return out2, attn_weights


class AggressiveFeatureAttention(nn.Module):
    """激进特征注意力 - 使用温度参数使注意力更集中"""
    def __init__(self, d_model, dropout=0.1, temperature=0.5):
        super(AggressiveFeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        attention_logits = self.attention(x)  # [batch*num_stocks, seq_len, 1]
        attention_weights = F.softmax(attention_logits / self.temperature, dim=1)
        attended = torch.sum(x * attention_weights, dim=1)  # [batch*num_stocks, d_model]
        return self.dropout(attended), attention_weights


class StockTransformer(nn.Module):
    """
    改进版StockTransformer选股模型
    
    改进点:
    1. 直接输出预期5日收益率 (而非排序分数)
    2. 添加不确定性估计头
    3. 使用GELU激活函数
    4. 增强的股票间交互注意力 (含前馈网络)
    5. 温度缩放的时序特征注意力
    """
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(StockTransformer, self).__init__()
        self.model_type = 'RankingTransformer'
        self.config = config
        self.num_stocks = num_stocks

        # 输入投影层
        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], config['sequence_length'])

        # 时序特征提取
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        # 特征注意力 (温度缩放版)
        feat_temp = config.get('feature_attention_temperature', 0.5)
        self.feature_attention = AggressiveFeatureAttention(config['d_model'], config['dropout'], temperature=feat_temp)

        # 股票间交互注意力 (增强版)
        self.cross_stock_attention = EnhancedCrossStockAttention(config['d_model'], config['nhead'], config['dropout'])

        # 排序特异性层 (使用GELU)
        self.ranking_layers = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.LayerNorm(config['d_model'] // 2),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )

        # 排序分数输出头 (无激活层 → 可以输出极端分数 → 真正激进)
        self.return_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.GELU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1)
            # 注意: 不加 Tanh! 让模型自由输出任意分数
        )

        # 不确定性估计头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.GELU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1),
            nn.Softplus()
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src, return_attention=False):
        # src: [batch, num_stocks, seq_len, feature_dim]
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        # 重塑为 [batch*num_stocks, seq_len, feature_dim]
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)

        # 输入投影和位置编码
        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)

        # 时序特征提取
        temporal_features = self.temporal_encoder(src_proj)

        # 特征注意力聚合
        aggregated_features, feat_attn_weights = self.feature_attention(temporal_features)

        # 重塑回股票维度
        stock_features = aggregated_features.view(batch_size, num_stocks, -1)

        # 股票间交互注意力
        interactive_features, cross_attn_weights = self.cross_stock_attention(stock_features)

        # 重塑回原形状
        interactive_features = interactive_features.view(batch_size * num_stocks, -1)

        # 排序特异性变换
        ranking_features = self.ranking_layers(interactive_features)

        # 生成排序分数 (无激活 → 可以极端表达 → 这才是真激进)
        ranking_scores = self.return_head(ranking_features)
        # ranking_scores: [batch*num_stocks, 1], 值域 (-∞, +∞)

        # 估计不确定性
        uncertainty = self.uncertainty_head(ranking_features)

        # 重塑
        ranking_scores = ranking_scores.view(batch_size, num_stocks)
        uncertainty = uncertainty.view(batch_size, num_stocks)

        if return_attention:
            return ranking_scores, uncertainty, feat_attn_weights, cross_attn_weights
        return ranking_scores, uncertainty
