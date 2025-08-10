# Hyperparameter Optimization for RL Trading

This module provides automatic hyperparameter optimization for the `n_step` parameter in the RL trading environment.

## Overview

The `n_step` parameter determines how many days after the current day are used to calculate the reward. Previously, this was hardcoded to 4 days. Now it's a configurable hyperparameter that can be automatically optimized to find the best value for your specific dataset.

## Key Features

- ✅ **Configurable n_step parameter**: No longer hardcoded to 4
- ✅ **Automatic optimization**: Finds the best n_step value automatically
- ✅ **Multiple metrics**: Optimize for Sharpe ratio, total return, or max drawdown
- ✅ **Configurable search space**: Customize which n_step values to test
- ✅ **Persistent results**: Saves optimization results for future use
- ✅ **Integration**: Training automatically uses optimized parameters
- 🚀 **Multiprocessing acceleration**: Parallel evaluation for faster optimization
- 🚀 **Flexible workers**: Auto-detect CPU cores or specify custom worker count

## Quick Start

### 1. Run Quick Test (Recommended First Step)
```bash
cd RL_moneyflow
python test_hyperopt.py
```

This runs a quick test with 3 n_step values (2, 4, 6) on 5 stocks to verify everything works.

### 2. Run Full Optimization (with multiprocessing)
```bash
python hyperopt.py                    # Auto-detect CPU cores
python hyperopt.py --workers 4        # Use 4 parallel workers
python hyperopt.py --sequential       # Disable parallel (use if issues)
```

This tests the full range of n_step values: [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]

### 3. Train with Optimized Parameters
```bash
python train.py --train
```

Training will automatically use the optimized n_step value from the optimization results.

### 4. Evaluate with Optimized Parameters
```bash
python train.py --eval
```

Evaluation will use the same optimized parameters.

## Configuration

### Default Configuration
The optimization uses these default settings:
- **n_step_range**: [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
- **optimization_metric**: sharpe_ratio
- **num_episodes**: 50 (for faster optimization)
- **subset_tickers**: 20 stocks (for speed)

### Custom Configuration
Create a custom config file:
```json
{
    "n_step_range": [1, 3, 5, 7],
    "gamma_range": [0.7],
    "batch_size_range": [64],
    "num_episodes": 100,
    "subset_tickers": 50,
    "metric": "sharpe_ratio"
}
```

Use it with:
```bash
python hyperopt.py --config my_config.json
```

## Command Line Options

### Basic Usage
```bash
# Use default settings with auto-detected workers
python hyperopt.py

# Custom n_step range
python hyperopt.py --n_step_range 1 2 3 4 5

# Different optimization metric
python hyperopt.py --metric total_return
python hyperopt.py --metric max_drawdown

# Custom config file
python hyperopt.py --config my_config.json
```

### Multiprocessing Options
```bash
# Auto-detect optimal worker count (default)
python hyperopt.py

# Specify exact number of workers
python hyperopt.py --workers 4
python hyperopt.py --workers 8

# Disable multiprocessing (sequential processing)
python hyperopt.py --sequential

# Combine with other options
python hyperopt.py --workers 6 --metric sharpe_ratio --n_step_range 1 3 5 7
```

## Files Generated

### Configuration Files
- `hyperopt_config.json`: Default optimization configuration
- `test_hyperopt_config.json`: Configuration for quick testing

### Results Files
- `hyperopt_results.json`: Complete optimization results including:
  - Best parameters found
  - All tested combinations
  - Performance metrics for each combination
  - Optimization settings used

## Understanding Results

### Best Parameters Example
```json
{
    "best_params": {
        "n_step": 6,
        "gamma": 0.7,
        "batch_size": 64,
        "score": 0.73
    },
    "optimization_metric": "sharpe_ratio"
}
```

This means `n_step=6` gave the highest Sharpe ratio of 0.73.

### Results Summary
The optimization will show:
1. **Best n_step value** and its performance
2. **Top 5 parameter combinations**
3. **Total combinations tested**
4. **Any errors encountered**

## Integration with Training

The training script (`train.py`) automatically:
1. Looks for `hyperopt_results.json`
2. Loads the best parameters if available
3. Falls back to defaults if no optimization results found
4. Uses optimized parameters for both training and evaluation

## Performance Considerations

### Multiprocessing Acceleration 🚀
- **Default**: Auto-detects CPU cores and uses all available
- **Speedup**: 2-8x faster depending on your CPU cores
- **Memory**: Each worker uses ~1-2 GB RAM
- **Recommended**: Use `--workers 4` for most systems

### Full Optimization Time
- **Sequential**: 50+ stocks × 11 n_step values = ~2-3 hours
- **Parallel (4 workers)**: ~30-45 minutes (2-4x speedup)
- **Parallel (8 workers)**: ~15-25 minutes (4-8x speedup)
- **Memory usage**: ~2-4 GB RAM per worker
- **Disk space**: ~500 MB for models (cleaned up automatically)

### Speed Optimization Tips
1. **Use multiprocessing**: `python hyperopt.py --workers 4` (fastest)
2. **Reduce subset_tickers**: Use fewer stocks for testing
3. **Reduce num_episodes**: Less training per combination  
4. **Reduce n_step_range**: Test fewer values
5. **Use quick test first**: Verify everything works with small test

### Worker Count Guidelines
- **Small tests**: `--workers 2` (balance speed vs resources)
- **Full optimization**: `--workers 4-8` (optimal for most systems)
- **High-end systems**: Auto-detect with no `--workers` flag
- **Resource-constrained**: `--sequential` or `--workers 1`

## Troubleshooting

### Common Issues

**"No data available"**
- Check if the data file exists: `data/refined_merged_moneyflow_ohlc_data.h5`
- Verify the date split in the config

**"CUDA out of memory"**
- Reduce `batch_size_range` in config
- Reduce `subset_tickers` 
- Use CPU instead of GPU in env.py

**"Module not found"**
- Run from the correct directory: `cd RL_moneyflow`
- Check Python path includes parent directories

### Debugging
Run the test script first to diagnose issues:
```bash
python test_hyperopt.py
```

This checks dependencies and runs a minimal optimization.

## Advanced Usage

### Cross-Validation
For more robust results, increase `cv_folds` in config:
```json
{
    "cv_folds": 3
}
```

### Multiple Metrics
You can analyze results for different metrics by re-running with different `--metric` options.

### Custom Search Spaces
Extend optimization to other hyperparameters by modifying `hyperopt.py`:
- Add `learning_rate_range`
- Add `memory_size_range`
- Add `target_update_range`

## What's New

### Changes Made
1. **env.py**: `n_step` parameter now configurable (was hardcoded to 4)
2. **train.py**: Automatically loads and uses optimized hyperparameters
3. **hyperopt.py**: New hyperparameter optimization module
4. **test_hyperopt.py**: Quick testing and verification script

### Backward Compatibility
- If no optimization results exist, uses original default (`n_step=4`)
- All existing functionality preserved
- No breaking changes to API

## Next Steps

After running optimization:
1. ✅ **Check results**: Review `hyperopt_results.json` 
2. ✅ **Train model**: Run `python train.py --train`
3. ✅ **Evaluate performance**: Run `python train.py --eval`
4. ✅ **Compare**: Compare performance with default n_step=4

## Example Workflow

```bash
# 1. Quick test to verify setup
python test_hyperopt.py

# 2. Run full optimization (takes 1-2 hours)
python hyperopt.py

# 3. Check results
cat hyperopt_results.json

# 4. Train with optimized parameters  
python train.py --train

# 5. Evaluate performance
python train.py --eval
```

The system will automatically use the best `n_step` value found during optimization!