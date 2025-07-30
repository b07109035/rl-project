# 強化學習交易系統 (RL Trading System)

基於 DDPG (Deep Deterministic Policy Gradient) 算法的量化交易系統，能夠學習最佳的交易策略並自動執行買賣決策。

## 📋 專案概述

本專案實現了一個完整的強化學習交易環境，包含：
- 自定義的交易環境 (TradingEnv)
- DDPG 深度強化學習智能體
- 多種交易策略選擇器
- 多樣化的獎勵函數設計

## 🏗️ 系統架構

```
rl-project/
├── DDPG-train.py    # 主要訓練腳本，包含環境、智能體和訓練邏輯
├── strategy.py      # 策略選擇器，處理動作轉換和獎勵計算
└── README.md        # 專案說明文件
```

## 🚀 主要功能

### 1. 交易環境 (TradingEnv)
- **狀態空間**: OHLCV (開盤價、最高價、最低價、收盤價、成交量)
- **動作空間**: 根據選定策略決定 (連續或離散)
- **獎勵機制**: 支援多種獎勵函數
- **資金管理**: 包含淨值、回撤、持倉管理

### 2. DDPG 智能體
- **Actor 網路**: 輸出交易動作
- **Critic 網路**: 評估動作價值
- **目標網路**: 軟更新機制提升訓練穩定性
- **經驗回放**: 提升學習效率

### 3. 交易策略

#### Strategy 1: 連續動作策略
- **動作維度**: 1
- **激活函數**: tanh
- **描述**: 根據 Actor 輸出的連續值決定買賣數量

#### Strategy 2: 離散動作策略  
- **動作維度**: 3
- **激活函數**: softmax
- **描述**: 三種動作選擇 (買入/賣出/持有)

#### Strategy 3: 待開發
- **狀態**: TODO

### 4. 獎勵函數

| 獎勵函數 | 描述 | 計算方式 |
|---------|------|----------|
| reward_function_1 | 動作收益 | `real_action × (next_price - current_price)` |
| reward_function_2 | 持倉收益 | `position_size × (next_price - current_price)` |
| reward_function_3 | 風險調整收益 | `return(%) + λ × drawdown(%)` |
| reward_function_4 | 夏普比率 | `mean(returns) / std(returns)` (使用過去64個數據點) |

## 🛠️ 安裝要求

```bash
pip install tensorflow pandas numpy gym
```

## 📊 關鍵參數設定

### 環境參數
- `initial_balance`: 初始資金 (預設: 10,000)
- `max_qty_per_order`: 每筆訂單最大數量 (預設: 10,000)
- `pyramiding`: 最大持倉倍數 (預設: 3)

### DDPG 參數
- `learning_rate`: 學習率 (預設: 1e-4)
- `batch_size`: 批次大小 (預設: 64)
- `gamma`: 折扣因子 (預設: 0.99)
- `tau`: 軟更新參數 (預設: 0.005)

### 訓練參數
- `episodes`: 訓練回合數 (預設: 50)
- `noise_scale`: 探索噪聲 (初始: 0.1，衰減率: 0.99)

## 💻 使用方式

### 1. 準備數據
確保有包含以下欄位的 CSV 檔案：
- `datetime`: 時間戳記
- `Open`: 開盤價
- `High`: 最高價  
- `Low`: 最低價
- `Close`: 收盤價
- `Volume`: 成交量

### 2. 執行訓練
```bash
python DDPG-train.py
```

### 3. 訓練配置
在 `train_ddpg()` 函數中修改以下設定：

```python
# 資料路徑
data = pd.read_csv('your_data_path.csv')

# 策略和獎勵函數組合
strategy_list = ['strategy_1', 'strategy_2']
reward_list = ['reward_function_1', 'reward_function_2', 'reward_function_4']
```

## 📈 輸出結果

訓練完成後會產生：
1. **訓練結果 CSV**: 包含每個時間步的交易記錄
   - 淨值變化
   - 收益率
   - 持倉大小
   - 回撤情況

2. **模型權重**: Actor 和 Critic 網路的訓練權重
   - `actor_{strategy}_{reward_function}`
   - `critic_{strategy}_{reward_function}`

## 🔧 自定義開發

### 新增交易策略
在 `strategy.py` 中的 `convert_to_real_action()` 方法中新增策略邏輯：

```python
if self.strategy_name == 'strategy_3':
    # 實現您的策略邏輯
    pass
```

### 新增獎勵函數
在 `calc_reward()` 方法中新增獎勵函數：

```python
if self.reward_function == 'reward_function_5':
    # 實現您的獎勵計算邏輯
    self.reward = your_reward_calculation
```

## ⚠️ 注意事項

1. **數據品質**: 確保輸入數據的完整性和準確性
2. **計算資源**: DDPG 訓練需要足夠的 GPU/CPU 資源
3. **過擬合風險**: 注意訓練和測試數據的分離
4. **風險管理**: 實際交易前請充分測試和驗證策略

## 📝 待辦事項

- [ ] 完成 Strategy 3 的實現
- [ ] 新增更多技術指標作為狀態特徵
- [ ] 實現金字塔式加倉 (pyramiding) 功能
- [ ] 新增模型評估和回測功能
- [ ] 優化獎勵函數設計

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request 來改進這個專案。

---

**免責聲明**: 本專案僅供學術研究和教育用途，不構成投資建議。實際交易存在風險，請謹慎使用。
