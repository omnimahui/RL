# RL Money Flow Trading System

A sophisticated reinforcement learning-based trading system that uses Deep Q-Networks (DQN) to make automated trading decisions based on money flow and OHLC (Open, High, Low, Close) market data.

## 🎯 Overview

This project implements a Deep Q-Learning agent that learns to make optimal trading decisions (buy, hold, sell) by analyzing market data including money flow indicators and traditional OHLC price data. The system features advanced hyperparameter optimization and parallel training capabilities for enhanced performance.

## ✨ Key Features

- **🤖 Deep Q-Network (DQN) Trading Agent**: Neural network-based decision making
- **📊 Money Flow Integration**: Advanced market sentiment analysis using money flow data
- **⚡ Hyperparameter Optimization**: Automated tuning for optimal performance
- **🚀 Parallel Processing**: Multi-core training and evaluation support
- **📈 Multiple Metrics**: Sharpe ratio, total return, and max drawdown optimization
- **🔧 Configurable Environment**: Flexible reward functions and trading parameters
- **💾 Model Persistence**: Save and load trained models
- **📋 Comprehensive Evaluation**: Detailed performance analytics

## 🏗️ Architecture

### Core Components

```
RL_moneyflow/
├── DQN/                    # Deep Q-Network implementation
│   ├── DeepQNetwork.py    # Neural network architecture
│   └── Train.py           # DQN training logic
├── env.py                 # Trading environment
├── train.py               # Main training script
├── train_parallel.py     # Parallel training implementation
├── hyperopt.py           # Hyperparameter optimization
├── Evaluation.py         # Performance evaluation
└── models/               # Saved trained models
```

### Neural Network Architecture

- **Input Layer**: Market features (OHLC + money flow indicators)
- **Hidden Layers**: 
  - Layer 1: 128 neurons with BatchNorm and ReLU
  - Layer 2: 256 neurons with BatchNorm and ReLU
- **Output Layer**: 3 actions (buy, hold, sell)

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch pandas numpy
```

### 1. Hyperparameter Optimization
Find the best parameters for your dataset:

```bash
# Auto-detect CPU cores for parallel processing
python hyperopt.py

# Use specific number of workers
python hyperopt.py --workers 4

# Sequential processing (if parallel has issues)
python hyperopt.py --sequential
```

### 2. Train Models
Train the DQN agent with optimized parameters:

```bash
python train.py --train
```

### 3. Evaluate Performance
Assess the trained model's performance:

```bash
python train.py --eval
```

## 📊 Data Requirements

The system expects refined merged money flow and OHLC data in HDF5 format:
- **File**: `../data/refined_merged_moneyflow_ohlc_data.h5`
- **Format**: Multi-index DataFrame with Date and Symbol levels
- **Features**: OHLC prices + money flow indicators

## ⚙️ Configuration

### Hyperparameter Optimization

Create a custom configuration file:

```json
{
    "n_step_range": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
    "gamma_range": [0.7, 0.8, 0.9],
    "batch_size_range": [64],
    "num_episodes": 50,
    "subset_tickers": 20,
    "metric": "sharpe_ratio",
    "cv_folds": 3,
    "max_workers": 4,
    "parallel": true
}
```

Use with:
```bash
python hyperopt.py --config my_config.json
```

### Environment Parameters

- **n_step**: Days ahead for reward calculation (optimizable)
- **gamma**: Discount factor for future rewards
- **batch_size**: Training batch size
- **transaction_cost**: Trading cost ratio
- **device**: CPU or CUDA for computation

## 📈 Performance Metrics

The system optimizes for:

1. **Sharpe Ratio**: Risk-adjusted returns (default)
2. **Total Return**: Absolute return performance
3. **Max Drawdown**: Risk management metric

## 🔧 Advanced Usage

### Custom Metrics

Modify the optimization metric:

```bash
python hyperopt.py --metric total_return
python hyperopt.py --metric max_drawdown
```

### Cross-Validation

Enable cross-validation for robust parameter selection:

```json
{
    "cv_folds": 5
}
```

## 📁 Output Files

### Configuration Files
- `hyperopt_config.json`: Default optimization settings
- `test_hyperopt_config.json`: Quick test configuration

### Results Files
- `hyperopt_results.json`: Complete optimization results
- `models/*.pkl`: Trained model files

### Evaluation Results
- Performance metrics and statistics
- Trading signal analysis
- Risk metrics calculation

## 🛠️ Troubleshooting

### Common Issues

**"No data available"**
- Verify data file exists: `../data/refined_merged_moneyflow_ohlc_data.h5`
- Check date range in configuration

**"CUDA out of memory"**
- Reduce batch size in configuration
- Switch to CPU in env.py: `device="cpu"`

**Performance Issues**
- Use parallel processing: `--workers 4`
- Reduce subset_tickers for testing
- Lower num_episodes for faster iteration

### Debugging

Run diagnostic tests:
```bash
python test_hyperopt.py          # Verify hyperparameter optimization
python debug_hyperopt.py         # Debug optimization issues
```

## 📊 Example Workflow

```bash
# 1. Verify setup
python test_hyperopt.py

# 2. Optimize hyperparameters (1-2 hours)
python hyperopt.py --workers 4

# 3. Check results
cat hyperopt_results.json

# 4. Train with optimal parameters
python train.py --train

# 5. Evaluate performance
python train.py --eval
```

## 🎯 Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| n_step | Reward calculation horizon | 4 | 1-15 |
| gamma | Discount factor | 0.7 | 0.1-0.99 |
| batch_size | Training batch size | 64 | 16-128 |
| num_episodes | Training episodes | 50 | 10-200 |

## 🔍 Performance Optimization

### Speed Improvements
1. **Use multiprocessing**: 2-8x faster training
2. **Optimize batch size**: Balance memory vs speed
3. **Reduce episode count**: Faster iteration during development
4. **GPU acceleration**: Enable CUDA if available

### Memory Management
- Each worker uses ~1-2 GB RAM
- Models automatically cleaned up after training
- ~500 MB disk space for temporary files

## 📚 Research Background

This system implements concepts from:
- Deep Q-Learning for financial markets
- Multi-step reward functions
- Money flow analysis in algorithmic trading
- Hyperparameter optimization for RL

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch for deep learning framework
- Pandas for data manipulation
- NumPy for numerical computations
- Hyperopt community for optimization techniques

---


**Note**: This system is for educational and research purposes. Always validate trading strategies thoroughly before using with real capital.
