import os
import sys
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
#import data.ohlc as ohlc
#import utils
import RL_moneyflow.env
from RL_moneyflow.DQN.Train import Train as DQN
import Evaluation
import pandas as pd
import numpy as np
# Import MODEL_FOLDER from local BaseTrain to avoid parent directory import
MODEL_FOLDER = f'{os.path.dirname(os.path.realpath(__file__))}/models'

def train_ticker_worker(args_tuple):
    """
    Worker function to train a single ticker model in parallel
    
    Args:
        args_tuple: (ticker, ticker_data_path, best_params, num_episodes)
    
    Returns:
        Dictionary with training results
    """
    ticker, best_params, num_episodes = args_tuple
    
    print(f"Worker: Training {ticker}...")
    
    try:
        # Load data for this specific ticker
        df_total = load_refined_merged_data()
        ticker_data = df_total[df_total.index.get_level_values(1) == ticker]
        
        if len(ticker_data) == 0:
            return {'ticker': ticker, 'error': 'No data available'}
        
        # Reset index for processing
        ticker_df = ticker_data.reset_index().set_index(['Date']).copy()
        
        # Process ticker training data
        data_train, df_close_train = data_process_single_ticker(ticker_df, type="train")
        
        if len(data_train) == 0:
            return {'ticker': ticker, 'error': 'No training data'}
        
        # Create training environment (use CPU for multiprocessing compatibility)
        train_env = RL_moneyflow.env.Env(
            data_train, 
            df_close_train, 
            action_name="BNS",
            device="cpu",  # Force CPU for multiprocessing
            n_step=best_params.get('n_step', 4),
            gamma=best_params.get('gamma', 0.7),
            batch_size=best_params.get('batch_size', 64)
        )
        
        # Create unique model name with process ID (but don't save individual models)
        model_name = f"{ticker}_pid{os.getpid()}_temp"
        
        # Train model for this ticker
        model = DQN(
            data_train, 
            train_env, 
            model_name,
            BATCH_SIZE=best_params.get('batch_size', 64),
            ReplayMemorySize=1000,
            TARGET_UPDATE=5,
            GAMMA=best_params.get('gamma', 0.7)
        )
        
        # Train the model (without saving individual models)
        model.train(num_episodes=num_episodes)
        
        print(f"Worker: Completed training {ticker}")
        
        return {
            'ticker': ticker,
            'model_name': model_name,
            'training_data_shape': data_train.shape,
            'success': True
        }
        
    except Exception as e:
        print(f"Worker: Error training {ticker}: {e}")
        return {
            'ticker': ticker,
            'error': str(e),
            'success': False
        }

def evaluate_ticker_worker(args_tuple):
    """
    Worker function to evaluate a single ticker in parallel
    
    Args:
        args_tuple: (ticker, best_params)
    
    Returns:
        Dictionary with evaluation results
    """
    ticker, best_params = args_tuple
    
    print(f"Worker: Evaluating {ticker}...")
    
    try:
        # Load data for this specific ticker
        df_total = load_refined_merged_data()
        ticker_data = df_total[df_total.index.get_level_values(1) == ticker]
        
        if len(ticker_data) == 0:
            return {'ticker': ticker, 'error': 'No data available'}
        
        # Reset index for processing
        ticker_df = ticker_data.reset_index().set_index(['Date']).copy()
        
        # Process ticker test data
        ticker_data_test, ticker_close = data_process_single_ticker(ticker_df, type="test")
        
        if len(ticker_data_test) == 0:
            return {'ticker': ticker, 'error': 'No test data'}
        
        # Print test data date range
        start_date = ticker_data_test.index.min()
        end_date = ticker_data_test.index.max()
        print(f"Worker: {ticker} test data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({len(ticker_data_test)} days)")
        
        # Create environment for evaluation (use CPU for multiprocessing compatibility)
        ticker_env = RL_moneyflow.env.Env(
            ticker_data_test, 
            ticker_close, 
            action_name="BNS",
            device="cpu",  # Force CPU for multiprocessing
            n_step=best_params.get('n_step', 4),
            gamma=best_params.get('gamma', 0.7),
            batch_size=best_params.get('batch_size', 64)
        )
        
        # Check if unified model exists
        unified_model_path = f"{MODEL_FOLDER}/unified_all_tickers-model.pkl"
        print (f"unified_model_path:{unified_model_path}")
        if not os.path.exists(unified_model_path):
            print(f"Worker: No unified model found at {unified_model_path}")
            return {'ticker': ticker, 'error': 'Unified model not found - run training first', 'success': False}
        
        # Use unified model for evaluation
        ticker_model = DQN(ticker_data_test, ticker_env, "unified_all_tickers")
        
        # Get actions
        ticker_actions = ticker_model.test(ticker_data_test)
        
        # Create evaluation dataframe
        ticker_eval_df = ticker_close.to_frame().rename(columns={"close": "close"}).reset_index().copy()
        
        # Convert numeric actions to string format
        action_mapping = {0: 'buy', 1: 'None', 2: 'sell'}
        actions_str = [action_mapping.get(action, 'None') for action in ticker_actions]
        
        # Ensure length matching
        min_length = min(len(ticker_eval_df), len(actions_str))
        ticker_eval_df = ticker_eval_df.iloc[:min_length].copy()
        ticker_eval_df['DQN_action'] = pd.Series(actions_str[:min_length], index=ticker_eval_df.index)
        ticker_eval_df.set_index(['Date'], inplace=True)
        
        # Evaluate performance
        ticker_ev_agent = Evaluation.Evaluation(ticker_eval_df, initial_investment=10000, trading_cost_ratio=0)
        
        # Get structured metrics from evaluate method
        metrics = ticker_ev_agent.evaluate()
        
        next_action = ticker_actions[-1] if ticker_actions else 'None'
        next_action_str = action_mapping.get(next_action, 'None')
        
        print(f"Worker: Completed evaluating {ticker}, next action: {next_action_str}")
        
        return {
            'ticker': ticker,
            'total_return': metrics.total_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'next_action': next_action_str,
            'test_data_shape': ticker_data_test.shape,
            'metrics': metrics.model_dump(),
            'success': True
        }
        
    except Exception as e:
        print(f"Worker: Error evaluating {ticker}: {e}")
        return {
            'ticker': ticker,
            'error': str(e),
            'success': False
        }

# UPDATED: Function to load refined merged data from China stocks 
# This replaces the original ohlc.load() to use the new merged dataset
def load_refined_merged_data():
    """
    Load the refined merged moneyflow and OHLC data for China stocks
    
    Returns DataFrame with combined data from:
    - moneyflow_data.h5: money flow indicators (net_amount, buy_lg_amount, etc.)  
    - ohlc_data.h5: price data (open, high, low, close, volume, amount)
    
    Data covers 5,152 China stocks from 2024-12-24 to 2025-08-08
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 
                            'data', 'refined_merged_moneyflow_ohlc_data.h5')
    print(f"Loading refined merged data from: {data_path}")
    df = pd.read_hdf(data_path, key='data')
    
    # Rename columns to match expected format
    df.rename(columns={
        'ts_code': 'symbol',
        'trade_date': 'Date',
        'vol': 'volume'
    }, inplace=True)
    
    # Set index to match expected format [Date, symbol]
    df.reset_index(inplace=True, drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(['Date', 'symbol'], inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Loaded {len(df):,} records for {df.index.get_level_values(1).nunique()} stocks")
    print(f"Date range: {df.index.get_level_values(0).min()} to {df.index.get_level_values(0).max()}")
    
    return df

# Use all available tickers from the dataset - will be populated dynamically
TICKERS = None
window_list = [5, 10,  20, 60]
split = '2025-07-01'  # Updated split date for China data range


def data_process_single_ticker(df, type="train"):
    """Process data for a single ticker"""
    # Handle column names for refined merged data
    # Drop columns that exist in the refined merged data
    columns_to_drop = []
    for col in ['name', 'latest','buy_lg_amount_rate','buy_md_amount','buy_md_amount_rate','buy_sm_amount','buy_sm_amount_rate']:
        if col in df.columns:
            columns_to_drop.append(col)
    
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)
    
    # Create percentage features from the merged data
    df['net_amount_pct'] = round(df['net_amount']/df['amount'] * 100)
    df['net_d5_amount_pct'] = round(df['net_d5_amount']/df['amount'] * 100)
    df['buy_lg_amount_pct'] = round(df['buy_lg_amount']/df['amount'] * 100)
    
    df_close = df['close']
    df = df.drop(['close', 'high', 'low', 'open', 'volume','amount', 'symbol'], axis=1)
    df.dropna(inplace=True)
    
    data_train = df[df.index < split]
    data_test = df[df.index >= split]
    df_close_train = df_close[df_close.index < split]
    df_close_test = df_close[df_close.index >= split]
    
    if type == "train":
        return data_train, df_close_train
    elif type == "test":
        return data_test, df_close_test

def data_process_all_tickers(df_total, type="train"):
    """Process data for all tickers, maintaining separate sequences"""
    all_data = []
    all_close = []
    all_tickers = []
    
    # Get unique tickers
    tickers = sorted(df_total.index.get_level_values(1).unique())
    
    for ticker in tickers:
        # Get data for this specific ticker
        ticker_data = df_total[df_total.index.get_level_values(1) == ticker]
        if len(ticker_data) == 0:
            continue
            
        # Reset index to have Date as main index for this ticker (make copy to avoid warnings)
        ticker_df = ticker_data.reset_index().set_index(['Date']).copy()
        
        # Process this ticker's data
        try:
            data, close = data_process_single_ticker(ticker_df, type=type)
            if len(data) > 0:  # Only add if we have data
                all_data.append(data)
                all_close.append(close)
                all_tickers.append(ticker)
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")
            continue
    
    if len(all_data) == 0:
        return pd.DataFrame(), pd.Series()
    
    # Concatenate all ticker data, but keep track of ticker boundaries
    combined_data = pd.concat(all_data, keys=all_tickers, names=['ticker', 'Date'])
    combined_close = pd.concat(all_close, keys=all_tickers, names=['ticker', 'Date'])
    
    return combined_data, combined_close
        
def load_best_hyperparams():
    """Load best hyperparameters from optimization results"""
    results_file = f"{os.path.dirname(os.path.realpath(__file__))}/hyperopt_results.json"
    
    # Default parameters
    default_params = {
        'n_step': 1,
        'gamma': 0.7,
        'batch_size': 64
    }
    
    if os.path.exists(results_file):
        try:
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            if results.get('best_params'):
                print(f"Using optimized hyperparameters from {results_file}")
                return results['best_params']
        except Exception as e:
            print(f"Error loading hyperparameters: {e}")
    
    print("Using default hyperparameters")
    return default_params

def train_parallel(max_workers=None, subset_size=20):
    """
    Train individual ticker models in parallel
    
    Args:
        max_workers: Number of parallel workers (None = auto-detect)
        subset_size: Number of tickers to train on
    """
    # Load optimized hyperparameters
    best_params = load_best_hyperparams()
    n_step = best_params.get('n_step', 1)
    gamma = best_params.get('gamma', 0.7)
    batch_size = best_params.get('batch_size', 64)
    
    print(f"Parallel training with n_step={n_step}, gamma={gamma}, batch_size={batch_size}")
    
    # Load data and get tickers
    df_total = load_refined_merged_data()
    all_tickers = sorted(df_total.index.get_level_values(1).unique())
    subset_tickers = all_tickers[:subset_size]
    
    print(f"Training {len(subset_tickers)} tickers in parallel")
    
    # Determine worker count
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(subset_tickers))
    
    print(f"Using {max_workers} parallel workers")
    
    # Prepare arguments for workers
    worker_args = [
        (ticker, best_params, 50)  # 50 episodes per ticker
        for ticker in subset_tickers
    ]
    
    # Train models in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all training tasks
        future_to_ticker = {
            executor.submit(train_ticker_worker, args): args[0] 
            for args in worker_args
        }
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_ticker)):
            try:
                result = future.result()
                results.append(result)
                
                if result.get('success'):
                    print(f"✅ Completed {result['ticker']} ({i+1}/{len(subset_tickers)})")
                else:
                    print(f"❌ Failed {result['ticker']}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                ticker = future_to_ticker[future]
                print(f"❌ Exception training {ticker}: {e}")
                results.append({'ticker': ticker, 'error': str(e), 'success': False})
    
    elapsed_time = time.time() - start_time
    
    # Summary
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    print(f"\n🏆 Parallel Training Summary:")
    print(f"   Time: {elapsed_time:.1f} seconds")
    print(f"   Successful: {len(successful)}/{len(results)}")
    print(f"   Failed: {len(failed)}")
    
    if failed:
        print(f"   Failed tickers: {[r['ticker'] for r in failed]}")
    
    return results

def train():
    # Load optimized hyperparameters
    best_params = load_best_hyperparams()
    n_step = best_params.get('n_step', 4)
    gamma = best_params.get('gamma', 0.7)
    batch_size = best_params.get('batch_size', 64)
    
    print(f"Training with n_step={n_step}, gamma={gamma}, batch_size={batch_size}")
    
    # Use refined merged data instead of ohlc.load()
    df_total = load_refined_merged_data()
    
    # Get all available tickers
    all_tickers = sorted(df_total.index.get_level_values(1).unique())
    print(f"Training unified model with all {len(all_tickers)} tickers")
    
    # Check if unified model already exists
    unified_model_path = f"{MODEL_FOLDER}/unified_all_tickers-model.pkl"
    print (f"unified_model_path:{unified_model_path}")
    if os.path.isfile(unified_model_path):
        print(f"Unified model already exists at {unified_model_path}")
        return  
    
    # Use smaller subset for faster training
    subset_tickers = all_tickers[:500]  # Reduced from all tickers
    df_total = df_total[df_total.index.get_level_values(1).isin(subset_tickers)]
    print(f"Using subset of {len(subset_tickers)} tickers for faster training")
    
    # Process all ticker data separately but combine for training
    data_train, df_close = data_process_all_tickers(df_total, type="train")
    
    if len(data_train) == 0:
        print("No training data available")
        return
    
    print(f"Training data shape: {data_train.shape}")
    print(f"Training on {data_train.index.get_level_values(0).nunique()} tickers")
    print(f"Data types are all numeric: {data_train.dtypes.apply(lambda x: str(x).startswith('float') or str(x).startswith('int')).all()}")
    
    # Create training environment with optimized parameters
    train_env = RL_moneyflow.env.Env(data_train, df_close, action_name="BNS", 
                          n_step=n_step, gamma=gamma, batch_size=batch_size)
    
    # Train one unified model with optimized parameters
    model = DQN(data_train, train_env, "unified_all_tickers", 
                BATCH_SIZE=batch_size,
                ReplayMemorySize=1000,    # Reasonable memory size
                TARGET_UPDATE=5,          # Original value
                GAMMA=gamma)
    
    # Train with reasonable episodes
    model.train(num_episodes=100)  # Back to original



def eval_parallel(max_workers=None, subset_size=20):
    """
    Evaluate multiple tickers in parallel
    
    Args:
        max_workers: Number of parallel workers (None = auto-detect)
        subset_size: Number of tickers to evaluate
    """
    # Load optimized hyperparameters
    best_params = load_best_hyperparams()
    n_step = best_params.get('n_step', 4)
    gamma = best_params.get('gamma', 0.7)
    batch_size = best_params.get('batch_size', 64)
    
    print(f"Parallel evaluation with n_step={n_step}, gamma={gamma}, batch_size={batch_size}")
    
    # Load data and get tickers
    df_total = load_refined_merged_data()
    all_tickers = sorted(df_total.index.get_level_values(1).unique())
    subset_tickers = all_tickers[:subset_size]
    
    print(f"Evaluating {len(subset_tickers)} tickers in parallel")
    
    # Determine worker count
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(subset_tickers))
    
    print(f"Using {max_workers} parallel workers")
    
    # Prepare arguments for workers
    worker_args = [
        (ticker, best_params)
        for ticker in subset_tickers
    ]
    
    # Evaluate models in parallel
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all evaluation tasks
        future_to_ticker = {
            executor.submit(evaluate_ticker_worker, args): args[0] 
            for args in worker_args
        }
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_ticker)):
            try:
                result = future.result()
                results.append(result)
                
                if result.get('success'):
                    metrics = result.get('metrics', {})
                    print(f"✅ {result['ticker']} ({i+1}/{len(subset_tickers)}):")
                    print(f"   Return: {result.get('total_return', 0):.2f}%, Sharpe: {result.get('sharpe_ratio', 0):.3f}")
                    print(f"   Volatility: {metrics.get('volatility', 0):.2f}, VaR: {metrics.get('value_at_risk', 0):.3f}")
                    print(f"   Next action: {result.get('next_action', 'None')}")
                else:
                    print(f"❌ Failed {result['ticker']}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                ticker = future_to_ticker[future]
                print(f"❌ Exception evaluating {ticker}: {e}")
                results.append({'ticker': ticker, 'error': str(e), 'success': False})
    
    elapsed_time = time.time() - start_time
    
    # Summary statistics
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]
    
    if successful:
        returns = [r.get('total_return', 0) for r in successful]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in successful]
        
        print(f"\n🏆 Parallel Evaluation Summary:")
        print(f"   Time: {elapsed_time:.1f} seconds")
        print(f"   Successful: {len(successful)}/{len(results)}")
        print(f"   Average return: {np.mean(returns):.2f}%")
        print(f"   Average Sharpe: {np.mean(sharpe_ratios):.3f}")
        print(f"   Best return: {max(returns):.2f}% ({[r['ticker'] for r in successful if r.get('total_return', 0) == max(returns)][0]})")
        
        # Show next actions summary
        actions = [r.get('next_action', 'None') for r in successful]
        action_counts = {action: actions.count(action) for action in set(actions)}
        print(f"   Next actions: {action_counts}")
    
    if failed:
        print(f"   Failed: {len(failed)} tickers")
    
    return results

def eval():
    # Load optimized hyperparameters
    best_params = load_best_hyperparams()
    n_step = best_params.get('n_step', 4)
    gamma = best_params.get('gamma', 0.7)
    batch_size = best_params.get('batch_size', 64)
    
    print(f"Evaluating with n_step={n_step}, gamma={gamma}, batch_size={batch_size}")
    
    # Use refined merged data instead of ohlc.load()
    df_total = load_refined_merged_data()
    
    # Get available tickers (same subset as training)
    all_tickers = sorted(df_total.index.get_level_values(1).unique())
    # Use same subset as training for consistency
    subset_tickers = all_tickers[:50]
    df_total = df_total[df_total.index.get_level_values(1).isin(subset_tickers)]
    
    print(f"Evaluating unified model on {len(subset_tickers)} individual stocks")
    
    # Evaluate each stock separately using the unified model
    for i, ticker in enumerate(subset_tickers):
        print(f"\n=== Stock {i+1}/{len(subset_tickers)}: {ticker} ===")
        
        # Get data for this specific ticker
        ticker_data = df_total[df_total.index.get_level_values(1) == ticker]
        if len(ticker_data) == 0:
            print(f"No data available for {ticker}")
            continue
            
        ticker_df = ticker_data.reset_index().set_index(['Date']).copy()
        
        # Process this ticker's test data
        try:
            ticker_data_test, ticker_close = data_process_single_ticker(ticker_df, type="test")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
            
        if len(ticker_data_test) == 0:
            print(f"No test data available for {ticker}")
            continue

        # Print test data date range
        start_date = ticker_data_test.index.min()
        end_date = ticker_data_test.index.max()
        print(f"Worker: {ticker} test data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({len(ticker_data_test)} days)")

        # Use unified model for individual ticker evaluation with optimized parameters
        ticker_env = RL_moneyflow.env.Env(ticker_data_test, ticker_close, action_name="BNS",
                               n_step=n_step, gamma=gamma, batch_size=batch_size)
        ticker_model = DQN(ticker_data_test, ticker_env, "unified_all_tickers")
        
        try:
            ticker_actions = ticker_model.test(ticker_data_test)
            
            # Create evaluation dataframe for this ticker (make explicit copy to avoid warnings)
            ticker_eval_df = ticker_close.to_frame().rename(columns={"close": "close"}).reset_index().copy()
            ticker_eval_df['DQN_action'] = pd.Series(ticker_actions)
            ticker_eval_df.set_index(['Date'], inplace=True)
            
            # Evaluate this ticker's performance
            ticker_ev_agent = Evaluation.Evaluation(ticker_eval_df, initial_investment=10000, trading_cost_ratio=0)
            ticker_ev_agent.evaluate()
            
            print(f"Next day action for {ticker}: {ticker_actions[-1] if ticker_actions else 'None'}")
            
        except Exception as e:
            print(f"Error evaluating {ticker}: {e}")
            continue
    

if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="Train models", action="store_true")
    parser.add_argument("-e", "--eval", help="Evaluate models", action="store_true")
    parser.add_argument("--parallel", help="Use parallel processing", action="store_true")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect)")
    parser.add_argument("--subset", type=int, default=20,
                       help="Number of tickers to process (default: 20)")
    
    if len(sys.argv) == 1:
        parser.print_usage()
        sys.exit(1)
        
    try:
        # Read arguments from command line
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    
    # Show parallel processing info
    if args.parallel:
        workers = args.workers if args.workers else mp.cpu_count()
        print(f"🚀 Parallel processing enabled: {workers} workers, {args.subset} tickers")
    
    if args.train:
        if args.parallel:
            train_parallel(max_workers=args.workers, subset_size=args.subset)
        else:
            train()
    elif args.eval:
        if args.parallel:
            eval_parallel(max_workers=args.workers, subset_size=args.subset)
        else:
            eval()