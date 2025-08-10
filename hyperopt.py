"""
Hyperparameter optimization for n_step parameter in RL trading environment
with multiprocessing support for acceleration
"""
import os
import sys
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Tuple, Any
import RL_moneyflow.env
from RL_moneyflow.DQN.Train import Train as DQN
import Evaluation

# Import data loading function from train.py
from train import load_refined_merged_data, data_process_all_tickers

MODEL_FOLDER = f'{os.path.dirname(os.path.realpath(__file__))}/models'

def evaluate_params_worker(params_tuple):
    """
    Worker function for multiprocessing evaluation
    This function runs in a separate process
    """
    n_step, gamma, batch_size, config = params_tuple
    
    print(f"Worker: Evaluating n_step={n_step}, gamma={gamma}, batch_size={batch_size}")
    
    try:
        # Load and prepare data (each process loads its own data)
        df_total = load_refined_merged_data()
        all_tickers = sorted(df_total.index.get_level_values(1).unique())
        
        # Use subset for faster optimization
        subset_tickers = all_tickers[:config["subset_tickers"]]
        df_total = df_total[df_total.index.get_level_values(1).isin(subset_tickers)]
        
        # Process training and test data
        data_train, df_close_train = data_process_all_tickers(df_total, type="train")
        data_test, df_close_test = data_process_all_tickers(df_total, type="test")
        
        if len(data_train) == 0 or len(data_test) == 0:
            return {
                'n_step': n_step,
                'gamma': gamma,
                'batch_size': batch_size,
                'error': 'No data available'
            }
        
        # Create environment with current hyperparameters
        train_env = RL_moneyflow.env.Env(
            data_train, 
            df_close_train, 
            action_name="BNS", 
            n_step=n_step,
            gamma=gamma,
            batch_size=batch_size
        )
        
        # Create unique model name for these parameters
        model_name = f"hyperopt_n{n_step}_g{gamma}_b{batch_size}_pid{os.getpid()}"
        
        # Train model
        model = DQN(
            data_train, 
            train_env, 
            model_name,
            BATCH_SIZE=batch_size,
            ReplayMemorySize=500,  # Reduced for faster training
            TARGET_UPDATE=5,
            GAMMA=gamma
        )
        
        # Train with reduced episodes for faster optimization
        model.train(num_episodes=config["num_episodes"])
        
        # Get actions from trained model
        test_actions = model.test(data_test)
        
        # Create evaluation dataframe
        eval_df = df_close_test.to_frame().copy()
        
        # Handle multi-level index properly
        if isinstance(eval_df.index, pd.MultiIndex):
            eval_df = eval_df.reset_index()
            eval_df = eval_df.set_index('Date')
        else:
            eval_df = eval_df.reset_index().set_index('Date' if 'Date' in eval_df.reset_index().columns else eval_df.reset_index().columns[0])
        
        # Ensure we don't have more actions than data points
        num_actions = min(len(test_actions), len(eval_df))
        eval_df = eval_df.iloc[:num_actions].copy()
        
        # Convert numeric actions to string format expected by Evaluation
        action_mapping = {0: 'buy', 1: 'None', 2: 'sell'}
        actions_to_add = [action_mapping.get(action, 'None') for action in test_actions[:num_actions]]
        eval_df['DQN_action'] = pd.Series(actions_to_add, index=eval_df.index)
        
        # Evaluate performance
        evaluator = Evaluation.Evaluation(
            eval_df, 
            initial_investment=10000, 
            trading_cost_ratio=0
        )
        
        # Capture evaluation results with better error handling
        import io
        from contextlib import redirect_stdout
        
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                evaluator.evaluate()
            output = f.getvalue()
            
            # Parse evaluation results
            metrics = parse_evaluation_output(output)
            
        except Exception as e:
            print(f"Worker evaluation error: {e}")
            metrics = {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'evaluation_error': str(e)
            }
        
        # Add hyperparameters to results
        metrics.update({
            'n_step': n_step,
            'gamma': gamma,
            'batch_size': batch_size
        })
        
        # Clean up model file
        model_path = f"{MODEL_FOLDER}/{model_name}-model.pkl"
        if os.path.exists(model_path):
            os.remove(model_path)
            
        print(f"Worker: Completed n_step={n_step}, score={metrics.get(config['metric'], 0):.4f}")
        return metrics
        
    except Exception as e:
        print(f"Worker error for n_step={n_step}: {e}")
        return {
            'n_step': n_step,
            'gamma': gamma,
            'batch_size': batch_size,
            'error': str(e)
        }

def parse_evaluation_output(output: str) -> Dict[str, float]:
    """Parse evaluation output to extract metrics"""
    metrics = {}
    
    lines = output.split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip().lower().replace(' ', '_')
                value_str = parts[1].strip()
                try:
                    # Try to extract numeric value
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', value_str)
                    if numbers:
                        value = float(numbers[0])
                        metrics[key] = value
                except:
                    pass
    
    # Set default values if not found
    if 'total_return' not in metrics:
        metrics['total_return'] = 0.0
    if 'sharpe_ratio' not in metrics:
        metrics['sharpe_ratio'] = 0.0
    if 'max_drawdown' not in metrics:
        metrics['max_drawdown'] = 0.0
        
    return metrics

class HyperparameterOptimizer:
    """
    Optimize n_step hyperparameter for RL trading environment
    """
    
    def __init__(self, config_file: str = "hyperopt_config.json"):
        self.config_file = config_file
        self.results = []
        self.best_params = None
        self.best_score = float('-inf')
        
        # Load or create default config
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load hyperparameter optimization configuration"""
        default_config = {
            "n_step_range": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
            "gamma_range": [0.7],  # Keep gamma fixed for now
            "batch_size_range": [64],  # Keep batch size fixed
            "num_episodes": 50,  # Reduced for faster optimization
            "train_test_split": "2025-07-01",
            "subset_tickers": 20,  # Use smaller subset for faster optimization
            "metric": "sharpe_ratio",  # Optimization metric
            "cv_folds": 3,  # For cross-validation
            "max_workers": None,  # Auto-detect CPU cores (None = use all cores)
            "parallel": True  # Enable multiprocessing
        }
        
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config_file)
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Merge with defaults for any missing keys
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
        else:
            config = default_config
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Created default config file: {config_path}")
        
        return config
    
    def save_config(self):
        """Save current configuration"""
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.config_file)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def evaluate_params(self, n_step: int, gamma: float = 0.7, batch_size: int = 64) -> Dict[str, float]:
        """
        Evaluate a specific parameter combination
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n=== Evaluating n_step={n_step}, gamma={gamma}, batch_size={batch_size} ===")
        
        try:
            # Load and prepare data
            df_total = load_refined_merged_data()
            all_tickers = sorted(df_total.index.get_level_values(1).unique())
            
            # Use subset for faster optimization
            subset_tickers = all_tickers[:self.config["subset_tickers"]]
            df_total = df_total[df_total.index.get_level_values(1).isin(subset_tickers)]
            
            # Process training data
            data_train, df_close_train = data_process_all_tickers(df_total, type="train")
            data_test, df_close_test = data_process_all_tickers(df_total, type="test")
            
            if len(data_train) == 0 or len(data_test) == 0:
                print("No training or test data available")
                return {"error": "No data"}
            
            # Create environment with current hyperparameters
            train_env = RL_moneyflow.env.Env(
                data_train, 
                df_close_train, 
                action_name="BNS", 
                n_step=n_step,
                gamma=gamma,
                batch_size=batch_size
            )
            
            # Create unique model name for these parameters
            model_name = f"hyperopt_n{n_step}_g{gamma}_b{batch_size}"
            
            # Train model
            model = DQN(
                data_train, 
                train_env, 
                model_name,
                BATCH_SIZE=batch_size,
                ReplayMemorySize=500,  # Reduced for faster training
                TARGET_UPDATE=5,
                GAMMA=gamma
            )
            
            # Train with reduced episodes for faster optimization
            model.train(num_episodes=self.config["num_episodes"])
            
            # Test on validation data
            test_env = RL_moneyflow.env.Env(
                data_test, 
                df_close_test, 
                action_name="BNS", 
                n_step=n_step,
                gamma=gamma,
                batch_size=batch_size
            )
            
            # Get actions from trained model
            test_actions = model.test(data_test)
            
            # Create evaluation dataframe
            eval_df = df_close_test.to_frame().copy()
            
            # Handle multi-level index properly
            if isinstance(eval_df.index, pd.MultiIndex):
                # Reset multi-level index and use Date as index
                eval_df = eval_df.reset_index()
                eval_df = eval_df.set_index('Date')
            else:
                # Single level index case
                eval_df = eval_df.reset_index().set_index('Date' if 'Date' in eval_df.reset_index().columns else eval_df.reset_index().columns[0])
            
            # Ensure we don't have more actions than data points
            num_actions = min(len(test_actions), len(eval_df))
            eval_df = eval_df.iloc[:num_actions].copy()  # Trim to match actions length
            
            # Convert numeric actions to string format expected by Evaluation
            action_mapping = {0: 'buy', 1: 'None', 2: 'sell'}
            actions_to_add = [action_mapping.get(action, 'None') for action in test_actions[:num_actions]]
            eval_df['DQN_action'] = pd.Series(actions_to_add, index=eval_df.index)
            
            # Evaluate performance
            evaluator = Evaluation.Evaluation(
                eval_df, 
                initial_investment=10000, 
                trading_cost_ratio=0
            )
            
            # Capture evaluation results with better error handling
            import io
            from contextlib import redirect_stdout
            
            try:
                f = io.StringIO()
                with redirect_stdout(f):
                    evaluator.evaluate()
                output = f.getvalue()
                
                # Parse evaluation results
                metrics = self.parse_evaluation_output(output)
                
            except ValueError as e:
                if "truth value of a Series is ambiguous" in str(e):
                    print(f"Series ambiguity error: {e}")
                    print("This might be due to action comparison in Evaluation module")
                    # Return minimal metrics to continue optimization
                    metrics = {
                        'total_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'evaluation_error': str(e)
                    }
                else:
                    raise
            except Exception as e:
                print(f"Evaluation error: {e}")
                metrics = {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'evaluation_error': str(e)
                }
            
            # Add hyperparameters to results
            metrics.update({
                'n_step': n_step,
                'gamma': gamma, 
                'batch_size': batch_size
            })
            
            # Clean up model file
            model_path = f"{MODEL_FOLDER}/{model_name}-model.pkl"
            if os.path.exists(model_path):
                os.remove(model_path)
                
            return metrics
            
        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            return {
                'n_step': n_step,
                'gamma': gamma,
                'batch_size': batch_size,
                'error': str(e)
            }
    
    def parse_evaluation_output(self, output: str) -> Dict[str, float]:
        """Parse evaluation output to extract metrics"""
        metrics = {}
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value_str = parts[1].strip()
                    try:
                        # Try to extract numeric value
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', value_str)
                        if numbers:
                            value = float(numbers[0])
                            metrics[key] = value
                    except:
                        pass
        
        # Set default values if not found
        if 'total_return' not in metrics:
            metrics['total_return'] = 0.0
        if 'sharpe_ratio' not in metrics:
            metrics['sharpe_ratio'] = 0.0
        if 'max_drawdown' not in metrics:
            metrics['max_drawdown'] = 0.0
            
        return metrics
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization with optional multiprocessing
        
        Returns:
            Best parameters and results
        """
        print("Starting hyperparameter optimization for n_step...")
        print(f"Testing n_step values: {self.config['n_step_range']}")
        
        # Generate parameter combinations
        param_combinations = []
        for n_step in self.config["n_step_range"]:
            for gamma in self.config["gamma_range"]:
                for batch_size in self.config["batch_size_range"]:
                    param_combinations.append((n_step, gamma, batch_size, self.config))
        
        print(f"Total parameter combinations to test: {len(param_combinations)}")
        
        # Determine number of workers
        if self.config.get("parallel", True):
            max_workers = self.config.get("max_workers", None)
            if max_workers is None:
                max_workers = min(mp.cpu_count(), len(param_combinations))
            
            print(f"Using parallel processing with {max_workers} workers")
            
            # Run parallel optimization
            start_time = time.time()
            
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_params = {
                    executor.submit(evaluate_params_worker, params): params 
                    for params in param_combinations
                }
                
                # Collect results as they complete
                for i, future in enumerate(as_completed(future_to_params)):
                    try:
                        result = future.result()
                        self.results.append(result)
                        
                        # Update best parameters if this is better
                        if 'error' not in result:
                            score = result.get(self.config["metric"], float('-inf'))
                            if score > self.best_score:
                                self.best_score = score
                                self.best_params = {
                                    'n_step': result['n_step'],
                                    'gamma': result['gamma'],
                                    'batch_size': result['batch_size'],
                                    'score': score
                                }
                                print(f"🏆 New best score: {score:.4f} with n_step={result['n_step']}")
                        
                        print(f"Progress: {i+1}/{len(param_combinations)} completed")
                        
                    except Exception as e:
                        params = future_to_params[future]
                        print(f"Error in worker for params {params[:3]}: {e}")
                        self.results.append({
                            'n_step': params[0],
                            'gamma': params[1], 
                            'batch_size': params[2],
                            'error': str(e)
                        })
            
            elapsed_time = time.time() - start_time
            print(f"Parallel optimization completed in {elapsed_time:.1f} seconds")
            
        else:
            print("Using sequential processing")
            
            # Sequential processing (original method)
            for i, (n_step, gamma, batch_size, _) in enumerate(param_combinations):
                print(f"\nProgress: {i+1}/{len(param_combinations)}")
                
                # Evaluate current parameters
                result = self.evaluate_params(n_step, gamma, batch_size)
                self.results.append(result)
                
                # Update best parameters if this is better
                if 'error' not in result:
                    score = result.get(self.config["metric"], float('-inf'))
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = {
                            'n_step': n_step,
                            'gamma': gamma,
                            'batch_size': batch_size,
                            'score': score
                        }
                        print(f"New best score: {score:.4f} with n_step={n_step}")
        
        # Save results
        self.save_results()
        
        return {
            'best_params': self.best_params,
            'all_results': self.results,
            'optimization_metric': self.config["metric"]
        }
    
    def save_results(self):
        """Save optimization results to file"""
        results_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 
            "hyperopt_results.json"
        )
        
        results_data = {
            'config': self.config,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'total_combinations_tested': len(self.results)
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=4, default=str)
        
        print(f"\nResults saved to: {results_file}")
    
    def print_summary(self):
        """Print optimization summary"""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*60)
        
        if self.best_params:
            print(f"Best n_step: {self.best_params['n_step']}")
            print(f"Best gamma: {self.best_params['gamma']}")
            print(f"Best batch_size: {self.best_params['batch_size']}")
            print(f"Best {self.config['metric']}: {self.best_score:.4f}")
        else:
            print("No successful optimization found")
        
        print(f"\nTotal combinations tested: {len(self.results)}")
        
        # Show top 5 results
        successful_results = [r for r in self.results if 'error' not in r]
        if successful_results:
            successful_results.sort(key=lambda x: x.get(self.config['metric'], float('-inf')), reverse=True)
            
            print(f"\nTop 5 results by {self.config['metric']}:")
            print("-" * 50)
            for i, result in enumerate(successful_results[:5]):
                print(f"{i+1}. n_step={result['n_step']}, "
                      f"{self.config['metric']}={result.get(self.config['metric'], 0):.4f}")

def main():
    """Main function to run hyperparameter optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for n_step")
    parser.add_argument("--config", default="hyperopt_config.json", 
                       help="Configuration file name")
    parser.add_argument("--metric", default="sharpe_ratio", 
                       choices=["sharpe_ratio", "total_return", "max_drawdown"],
                       help="Optimization metric")
    parser.add_argument("--n_step_range", nargs='+', type=int, 
                       default=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
                       help="Range of n_step values to test")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect)")
    parser.add_argument("--sequential", action="store_true",
                       help="Disable multiprocessing and run sequentially")
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(config_file=args.config)
    
    # Update config with command line args
    if args.metric:
        optimizer.config["metric"] = args.metric
    if args.n_step_range:
        optimizer.config["n_step_range"] = args.n_step_range
    if args.workers is not None:
        optimizer.config["max_workers"] = args.workers
    if args.sequential:
        optimizer.config["parallel"] = False
        print("Multiprocessing disabled - using sequential processing")
    
    # Save updated config
    optimizer.save_config()
    
    # Run optimization
    results = optimizer.optimize()
    
    # Print summary
    optimizer.print_summary()
    
    return results

if __name__ == "__main__":
    # Set multiprocessing start method for better cross-platform compatibility
    mp.set_start_method('spawn', force=True)
    results = main()