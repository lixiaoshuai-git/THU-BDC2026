import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_vectorized
import joblib
import os
import json
import multiprocessing as mp
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


feature_cloums_map = {
    '39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ],
    '158+39': [
        'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
        'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
        'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
        'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
        'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
        'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
        'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
        'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
        'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
        'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
        'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
        'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
        'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
        'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
        'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
        'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
        'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
        'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread',
        # 持有期收益相关特征 (新增)
        'vol_5d', 'vol_20d', 'vol_ratio_5_20',
        'momentum_5d', 'momentum_20d', 'momentum_ratio_5_20',
        'volume_change_5d', 'volume_ma_5d', 'volume_ma_20d', 'volume_ma_ratio_5_20',
        'ma_alignment',
        'up_count_5d', 'extreme_up_5d', 'extreme_down_5d',
        'return_skew_20d',
        'max_drawdown_5d',
    ]
}

feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}


def _build_label_and_clean(processed, drop_small_open=True):
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]
    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# ============================================================
# 新增: 直接优化组合收益率的损失函数
# ============================================================
class PortfolioReturnLoss(nn.Module):
    """直接优化组合收益率 + 不确定性加权"""
    def __init__(self, top_k=5, return_weight=1.0, risk_penalty=0.1):
        super(PortfolioReturnLoss, self).__init__()
        self.top_k = top_k
        self.return_weight = return_weight
        self.risk_penalty = risk_penalty

    def forward(self, predicted_returns, true_returns, uncertainties=None):
        # predicted_returns: [batch, num_stocks]
        # true_returns: [batch, num_stocks]
        batch_size = predicted_returns.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            _, top_k_indices = torch.topk(predicted_returns[i], self.top_k)
            portfolio_true_return = true_returns[i][top_k_indices].mean()
            portfolio_pred_return = predicted_returns[i][top_k_indices].mean()
            return_loss = F.mse_loss(portfolio_pred_return, portfolio_true_return)

            # 可选: 风险惩罚
            if self.risk_penalty > 0:
                portfolio_risk = torch.std(true_returns[i][top_k_indices])
                return_loss = return_loss + self.risk_penalty * portfolio_risk

            # 可选: 不确定性惩罚 (避免模型过度自信)
            if uncertainties is not None:
                top_uncertainty = uncertainties[i][top_k_indices].mean()
                return_loss = return_loss + 0.01 * top_uncertainty

            total_loss = total_loss + return_loss

        return total_loss / batch_size


# ============================================================
# 保留原排序损失 (兼容模式)
# ============================================================
class WeightedRankingLoss(nn.Module):
    def __init__(self, temperature=1.0, k=5, weight_factor=2.0, pairwise_weight=1, base_weight=1.0):
        super(WeightedRankingLoss, self).__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight

    def listwise_loss(self, y_pred, y_true, weights):
        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)
        weighted_ce = -(target_probs * torch.log(pred_probs + 1e-12) * weights)
        ce_loss = (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()
        return ce_loss

    def pairwise_loss(self, y_pred, y_true, weights):
        batch_size, num_items = y_pred.size()
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        mask = (true_diff != 0).float()
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        weighted_loss = pairwise_loss * mask * weight_matrix
        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        loss = (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()
        return loss

    def forward(self, y_pred, y_true):
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)
        _, top_indices = torch.topk(y_true, k, dim=1)
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor
        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)
        return listwise + self.pairwise_weight * pairwise


def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """原有的排序指标 (兼容)"""
    batch_size = y_pred.size(0)
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        if valid_indices.numel() < k:
            continue
        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()
        random_return_sum = k * valid_true.mean().item()
        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)

    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
        'ratio_pred': np.mean(ratio_pred_list) if ratio_pred_list else 0.0,
        'ratio_random': np.mean(ratio_random_list) if ratio_random_list else 0.0,
        'final_score': np.mean(final_score_list) if final_score_list else 0.0,
    }
    return metrics


# ============================================================
# 新增: 直接评估组合绩效的指标
# ============================================================
def calculate_portfolio_metrics(predicted_returns, true_returns, masks, k=5):
    batch_size = predicted_returns.size(0)
    pred_returns = []
    max_returns = []
    rand_returns = []

    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        if valid_indices.numel() < k:
            continue

        valid_pred = predicted_returns[i][valid_indices]
        valid_true = true_returns[i][valid_indices]

        # 预测组合
        _, pred_indices = torch.topk(valid_pred, k)
        pred_ret = valid_true[pred_indices].mean().item()

        # 最优组合
        _, true_indices = torch.topk(valid_true, k)
        max_ret = valid_true[true_indices].mean().item()

        # 随机组合
        rand_ret = valid_true.mean().item()

        pred_returns.append(pred_ret)
        max_returns.append(max_ret)
        rand_returns.append(rand_ret)

    n = len(pred_returns)
    if n == 0:
        return {'avg_portfolio_return': 0.0, 'final_score': 0.0, 'avg_max_return': 0.0}

    p = np.mean(pred_returns)
    m = np.mean(max_returns)
    r = np.mean(rand_returns)

    return {
        'avg_portfolio_return': p,
        'avg_max_return': m,
        'avg_random_return': r,
        'ratio_pred': p / (m + 1e-12),
        'ratio_random': r / (m + 1e-12),
        'final_score': (p - r) / (m - r + 1e-12),
    }


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),
            'targets': torch.FloatTensor(self.targets[idx]),
            'relevance': torch.LongTensor(self.relevance_scores[idx]),
            'stock_indices': torch.LongTensor(self.stock_indices[idx])
        }


def collate_fn(batch):
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]

    max_stocks = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []

    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, torch.zeros(pad_size)], dim=0)
            rel = torch.cat([rel, torch.zeros(pad_size, dtype=torch.long)], dim=0)
            stock_idx = torch.cat([stock_idx, torch.zeros(pad_size, dtype=torch.long)], dim=0)

        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)

    return {
        'sequences': torch.stack(padded_sequences),
        'targets': torch.stack(padded_targets),
        'relevance': torch.stack(padded_relevance),
        'stock_indices': torch.stack(padded_stock_indices),
        'masks': torch.stack(masks)
    }


def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer, accumulation_steps=1):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    optimizer.zero_grad()
    direct_return = config.get('direct_return_optimization', False)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)

        # 模型预测 (新: 同时返回预期收益率和不确定性)
        expected_returns, uncertainty = model(sequences)
        # expected_returns: [batch, max_stocks]
        # uncertainty: [batch, max_stocks]

        # 应用mask
        masked_returns = expected_returns * masks + (1 - masks) * (-1e9)
        masked_targets = targets * masks

        # 计算损失
        batch_loss = None
        batch_size = sequences.size(0)

        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()
            if valid_indices.numel() == 0:
                continue
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)

            valid_pred = masked_returns[i][valid_indices]
            valid_true = masked_targets[i][valid_indices]

            # direct_return 模式下需要至少有 k 只有效股票才能计算 top-k
            min_required = criterion.top_k if direct_return else 2
            if len(valid_pred) >= min_required:
                if direct_return:
                    # 直接优化组合收益率
                    loss = criterion(
                        valid_pred.unsqueeze(0),
                        valid_true.unsqueeze(0),
                        uncertainty[i][valid_indices].unsqueeze(0)
                    )
                else:
                    # 排序损失 (兼容模式)
                    _, sorted_indices = torch.sort(valid_true, descending=True)
                    relevance_scores = torch.zeros_like(valid_true, requires_grad=False)
                    relevance_scores[sorted_indices] = torch.arange(len(valid_true), 0, -1, device=device, dtype=torch.float32)
                    relevance_scores = relevance_scores.detach()
                    loss = criterion(valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0))

                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss

        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss = batch_loss / accumulation_steps  # 梯度累积缩放
            batch_loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch * len(dataloader) + local_step)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += batch_loss.item() * accumulation_steps

            with torch.no_grad():
                if direct_return:
                    metrics = calculate_portfolio_metrics(masked_returns, masked_targets, masks, k=5)
                else:
                    metrics = calculate_ranking_metrics(masked_returns, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v

            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item() * accumulation_steps, global_step=epoch * len(dataloader) + local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch * len(dataloader) + local_step)

    # 处理最后不完整的梯度累积批次
    batch_iterated = len(dataloader) > 0
    if batch_iterated and local_step > 0 and (len(dataloader) % accumulation_steps != 0):
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        optimizer.step()
        optimizer.zero_grad()

    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics


def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    direct_return = config.get('direct_return_optimization', False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            expected_returns, uncertainty = model(sequences)
            masked_returns = expected_returns * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

            batch_loss = None
            batch_size = sequences.size(0)

            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                if valid_indices.numel() == 0:
                    continue
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)

                valid_pred = masked_returns[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]

                # direct_return 模式下需要至少有 k 只有效股票
                min_required = criterion.top_k if direct_return else 2
                if len(valid_pred) >= min_required:
                    if direct_return:
                        loss = criterion(
                            valid_pred.unsqueeze(0),
                            valid_true.unsqueeze(0),
                            uncertainty[i][valid_indices].unsqueeze(0)
                        )
                    else:
                        _, sorted_indices = torch.sort(valid_true, descending=True)
                        relevance_scores = torch.zeros_like(valid_true, requires_grad=False)
                        relevance_scores[sorted_indices] = torch.arange(len(valid_true), 0, -1, device=device, dtype=torch.float32)
                        relevance_scores = relevance_scores.detach()
                        loss = criterion(valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0))

                    batch_loss = batch_loss + loss if batch_loss is not None else loss

            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()

            if direct_return:
                metrics = calculate_portfolio_metrics(masked_returns, masked_targets, masks, k=5)
            else:
                metrics = calculate_ranking_metrics(masked_returns, masked_targets, masks, k=5)
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches

    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)

    return avg_loss, total_metrics


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    model.eval()
    latest_date = data['日期'].max()

    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []

    for stock_code in data['股票代码'].unique():
        stock_history = data[
            (data['股票代码'] == stock_code) &
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])

    if len(day_sequences) == 0:
        return []

    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)

    with torch.no_grad():
        expected_returns, uncertainty = model(sequences)
        scores = expected_returns.squeeze().cpu().numpy()  # 直接使用预期收益率作为分数
        uncertainty = uncertainty.squeeze().cpu().numpy()

        top_indices = np.argsort(scores)[::-1][:top_k]

        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_return': scores[idx],
                'uncertainty': uncertainty[idx],
                'rank': len(top_stocks) + 1
            })

    return top_stocks


def save_predictions(top_stocks, output_path):
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测收益率': stock.get('predicted_return', stock.get('predicted_score', 0)),
            '不确定性': stock.get('uncertainty', 0)
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def split_train_val_by_last_month(df, sequence_length):
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()

    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后一个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


def get_scheduler(optimizer, num_epochs, train_loader_len):
    """获取学习率调度器"""
    scheduler_type = config.get('lr_scheduler', 'cosine_restart')
    if scheduler_type == 'cosine_restart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('cosine_T_0', 10),
            T_mult=config.get('cosine_T_mult', 2),
            eta_min=config.get('cosine_eta_min', 1e-6)
        )
    elif scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'] * 10,
            total_steps=num_epochs * train_loader_len,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.2,
            total_iters=num_epochs
        )
    return scheduler


def main():
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    data_path = config['data_path']
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"使用设备: {device}")
    print(f"优化模式: {'直接优化组合收益率' if config.get('direct_return_optimization', False) else '排序优化'}")

    # 1. 数据加载
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    # 2. 特征工程与预处理
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    # 3. 标准化
    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # 4. 创建排序数据集
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data, features, config['sequence_length'],
        ranking_data_path=config.get('train_ranking_data_path')
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data, features, config['sequence_length'],
        ranking_data_path=config.get('val_ranking_data_path'),
        min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"验证集样本数: {len(val_sequences)}")

    # 5. 数据加载器
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    # 6. 模型初始化
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 7. 损失函数
    direct_return = config.get('direct_return_optimization', False)
    if direct_return:
        criterion = PortfolioReturnLoss(
            top_k=5,
            return_weight=config.get('return_weight', 1.0),
            risk_penalty=config.get('risk_penalty', 0.1)
        )
        print("使用 PortfolioReturnLoss (直接优化组合收益率)")
    else:
        criterion = WeightedRankingLoss(
            temperature=config.get('loss_temperature', 0.5),
            k=5,
            weight_factor=config.get('top5_weight', 5.0),
            pairwise_weight=config.get('pairwise_weight', 2.0),
            base_weight=config.get('base_weight', 1.0)
        )
        print("使用 WeightedRankingLoss (排序优化)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-5))

    scheduler_type = config.get('lr_scheduler', 'cosine_restart')
    scheduler = get_scheduler(optimizer, config['num_epochs'], len(train_loader))
    print(f"学习率调度器: {scheduler_type}")

    # 8. 训练
    accumulation_steps = config.get('gradient_accumulation_steps', 1)
    if accumulation_steps > 1:
        print(f"梯度累积步数: {accumulation_steps} (有效批次大小 = {config['batch_size'] * accumulation_steps})")

    if is_train:
        best_score = -float('inf')
        best_epoch = -1

        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")

            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer,
                accumulation_steps=accumulation_steps
            )

            score_name = 'avg_portfolio_return' if direct_return else 'pred_return_sum'
            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")

            eval_loss, eval_metrics = evaluate_ranking_model(
                model, val_loader, criterion, device, writer, epoch
            )

            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.4f}")

            # 学习率调度
            if scheduler_type == 'onecycle':
                # OneCycle需要在每个batch后调用
                pass
            else:
                scheduler.step()

            if writer:
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0] if not scheduler_type == 'onecycle' else optimizer.param_groups[0]['lr'], global_step=epoch)

            # 保存最佳模型
            current_final_score = eval_metrics.get('final_score', 0.0)
            if current_final_score > best_score:
                best_score = current_final_score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"保存最佳模型 - final_score: {best_score:.4f}")

        print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 final_score: {best_score:.4f}")
        with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
            f.write(f"Best epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")

        if writer:
            writer.close()

        return best_score


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 final_score: {best_score:.4f} ##########")
