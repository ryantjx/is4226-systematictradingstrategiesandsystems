import pandas as pd
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
"""
Key Functions
# Assume 'df_strategy' and 'df_benchmark' are your prepared DataFrames
performance, trades = get_performance(df_strategy, df_benchmark, 
                                      strategy_signal_column='signal', 
                                      strategy_returns_column='returns_close')
print(performance)
plot_performance(df_strategy, df_benchmark, trades, 
                 strategy_price_column='Close', 
                 strategy_returns_column='returns_strategy', 
                 benchmark_returns_column='returns_close')
"""

def get_trades(df_strategy, strategy_signal_column='signal', strategy_price_column='Close'):
    if df_strategy[strategy_signal_column].mean() == 0:
        print("No trades executed")
        return None, {}
    
    df_strategy['signal_change'] = df_strategy[strategy_signal_column] != df_strategy[strategy_signal_column].shift(1)
    if df_strategy['signal_change'].mean() == 0:
        print("No trades executed")
        return None, {}
    
    trades = []
    open_trade = None

    for idx, row in df_strategy[df_strategy['signal_change']].iterrows():
        current_signal = row[strategy_signal_column]
        
        if open_trade is None:
            if current_signal != 0:
                open_trade = {
                    'entry_datetime': datetime.fromtimestamp(row['timestamp_utc'] / 1000, tz=timezone.utc),
                    'entry_price': row[strategy_price_column],
                    'position': current_signal
                }
        else:
            exit_datetime = datetime.fromtimestamp(row['timestamp_utc'] / 1000, tz=timezone.utc)
            duration = exit_datetime - open_trade['entry_datetime']
            exit_price = row[strategy_price_column]
            pnl = (exit_price - open_trade['entry_price']) * open_trade['position']
            pnl_pct = (pnl / abs(open_trade['entry_price'])) * 100
            
            trades.append({
                'entry_datetime': open_trade['entry_datetime'],
                'exit_datetime': exit_datetime,
                'duration': duration,
                'entry_price': open_trade['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl%': pnl_pct,
                'position': open_trade['position'],
                'win': 1 if pnl > 0 else 0
            })
            
            if current_signal != 0:
                open_trade = {
                    'entry_datetime': exit_datetime,
                    'entry_price': exit_price,
                    'position': current_signal
                }
            else:
                open_trade = None
    
    if open_trade is not None:
        last_row = df_strategy.iloc[-1]
        exit_datetime = datetime.fromtimestamp(last_row['timestamp_utc'] / 1000, tz=timezone.utc)
        duration = exit_datetime - open_trade['entry_datetime']
        exit_price = last_row[strategy_price_column]
        pnl = (exit_price - open_trade['entry_price']) * open_trade['position']
        pnl_pct = (pnl / abs(open_trade['entry_price'])) * 100
        
        trades.append({
            'entry_datetime': open_trade['entry_datetime'],
            'exit_datetime': exit_datetime,
            'duration': duration,
            'entry_price': open_trade['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl%': pnl_pct,
            'position': open_trade['position'],
            'win': 1 if pnl > 0 else 0
        })
    
    trades_df = pd.DataFrame(trades)
    
    performance = calculate_trade_performance(trades_df)
    
    return trades_df, performance

def calculate_trade_performance(trades_df):
    num_trades = len(trades_df)
    win_rate = round(trades_df['win'].mean(), 5)
    avg_win = trades_df[trades_df['win'] == 1]['pnl'].mean()
    avg_loss = abs(trades_df[trades_df['win'] == 0]['pnl'].mean())
    reward_risk_ratio = avg_win / avg_loss if avg_loss != 0 else np.inf
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    avg_duration = trades_df['duration'].mean()
    
    return {
        'number_of_trades': num_trades,
        'win_rate': win_rate,
        'reward_risk_ratio': reward_risk_ratio,
        'expectancy': expectancy,
        'avg_trade_duration': avg_duration,
    }

def get_performance(df_strategy, df_benchmark, risk_free_rate=0, strategy_signal_column='signal', strategy_price_column='Close', strategy_returns_column='returns_close'):
    df_strategy['returns_strategy'] = df_strategy[strategy_signal_column].shift(1) * df_strategy[strategy_returns_column]
    df_strategy['logreturns_strategy'] = df_strategy[strategy_signal_column].shift(1) * np.log(1 + df_strategy[strategy_returns_column])

    performance = {}
    trades, trades_performance = get_trades(df_strategy, strategy_signal_column=strategy_signal_column, strategy_price_column=strategy_price_column)
    performance.update(trades_performance)
    performance.update(calculate_relative_metrics(df_strategy, df_benchmark, risk_free_rate))
    performance.update(calculate_confidence_metrics(df_strategy))

    return performance, trades

def calculate_relative_metrics(df_strategy, df_benchmark, risk_free_rate):
    beta = calculate_beta(df_strategy['returns_strategy'], df_benchmark['returns_close'])
    alpha = calculate_alpha(beta, df_strategy, 'returns_strategy', df_benchmark, 'returns_close', risk_free_rate)
    
    annualized_sharpe = ((calculate_annualized_return(df_strategy, 'returns_strategy') - risk_free_rate) / df_strategy['returns_strategy'].std()) * np.sqrt(252)
    
    downside_returns = df_strategy['returns_strategy'][df_strategy['returns_strategy'] < 0]
    downside_std = downside_returns.std()
    annualized_sortino = ((calculate_annualized_return(df_strategy, 'returns_strategy') - risk_free_rate) / downside_std) * np.sqrt(252)
    
    cumulative_returns = (1 + df_strategy['returns_strategy']).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    annualized_calmar = calculate_annualized_return(df_strategy, 'returns_strategy') / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    annualized_treynor = (calculate_annualized_return(df_strategy, 'returns_strategy') - risk_free_rate) / beta
    annualized_information = (calculate_annualized_return(df_strategy, 'returns_strategy') - calculate_annualized_return(df_benchmark, 'returns_close')) / (df_strategy['returns_strategy'] - df_benchmark['returns_close']).std()
    
    omega_ratio = np.mean(np.maximum(df_strategy['returns_strategy'] - risk_free_rate, 0)) / np.mean(np.maximum(risk_free_rate - df_strategy['returns_strategy'], 0)) if np.mean(np.maximum(risk_free_rate - df_strategy['returns_strategy'], 0)) != 0 else (np.inf if np.mean(np.maximum(df_strategy['returns_strategy'] - risk_free_rate, 0)) > 0 else 0)
    
    profit_factor = df_strategy['returns_strategy'][df_strategy['returns_strategy'] > 0].sum() / -df_strategy['returns_strategy'][df_strategy['returns_strategy'] < 0].sum() if -df_strategy['returns_strategy'][df_strategy['returns_strategy'] < 0].sum() != 0 else np.inf
    recovery_factor = ((1 + df_strategy['returns_strategy']).prod() - 1) / abs(max_drawdown) if max_drawdown != 0 else np.inf
    
    return {
        "beta": beta,
        "alpha": alpha,
        "max_drawdown" : max_drawdown,
        "annualized_sharpe": annualized_sharpe,
        "annualized_sortino": annualized_sortino,
        "annualized_calmar": annualized_calmar,
        "annualized_treynor": annualized_treynor,
        "annualized_information": annualized_information,
        "omega_ratio": omega_ratio,
        "profit_factor": profit_factor,
        "recovery_factor": recovery_factor
    }

def calculate_confidence_metrics(df_strategy, confidence=0.95):
    var = np.percentile(df_strategy['returns_strategy'], 100 * (1-confidence))
    cvar = df_strategy['returns_strategy'][df_strategy['returns_strategy'] <= var].mean()
    return {
        "VaR": var,
        "CVaR": cvar
    }

def calculate_beta(returns_strategy, returns_benchmark):
    covariance = np.cov(returns_strategy.dropna(), returns_benchmark.dropna())[0, 1]
    benchmark_variance = np.var(returns_benchmark.dropna())
    return covariance / benchmark_variance if benchmark_variance != 0 else np.nan

def calculate_alpha(beta, df_strategy, strategy_column, df_benchmark, benchmark_column, risk_free_rate):
    annualized_returns_strategy = calculate_annualized_return(df_strategy, strategy_column)
    annualized_returns_benchmark = calculate_annualized_return(df_benchmark, benchmark_column)
    return annualized_returns_strategy - (risk_free_rate + beta * (annualized_returns_benchmark - risk_free_rate))

def calculate_annualized_return(dataframe, column_name):
    total_return = (1 + dataframe[column_name]).prod() - 1
    n_years = len(dataframe) / 252  # Assuming 252 trading days in a year
    return (1 + total_return) ** (1 / n_years) - 1

def plot_performance(df_strategy, df_benchmark, trades, strategy_price_column='Close', strategy_returns_column='returns_strategy', benchmark_returns_column='returns_close'):
    df_strategy['datetime'] = pd.to_datetime(df_strategy['timestamp_utc'], unit='ms', utc=True)
    df_benchmark['datetime'] = pd.to_datetime(df_benchmark['timestamp_utc'], unit='ms', utc=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 18))
    
    # Plot 1: Trades and Price
    ax1.plot(df_strategy['datetime'], df_strategy[strategy_price_column], label=strategy_price_column)
    ax1.set_title('Trades and Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    
    for _, trade in trades.iterrows():
        color = 'green' if trade['position'] == 1 else 'red'
        ax1.axvspan(trade['entry_datetime'], trade['exit_datetime'], color=color, alpha=0.1)
        
        marker = '^' if trade['win'] == 1 else 'v'
        marker_color = 'green' if trade['win'] == 1 else 'red'
        ax1.plot(trade['exit_datetime'], 
                df_strategy.loc[df_strategy['datetime'] == trade['exit_datetime'], strategy_price_column].values[0],
                marker=marker, color=marker_color, markersize=10)
    
    ax1.legend()
    
    # Plot 2: Returns Comparison
    cumulative_strategy = (1 + df_strategy[strategy_returns_column]).cumprod()
    cumulative_benchmark = (1 + df_benchmark[benchmark_returns_column]).cumprod()
    
    ax2.plot(df_strategy['datetime'], cumulative_strategy, label='Strategy')
    ax2.plot(df_benchmark['datetime'], cumulative_benchmark, label='Benchmark')
    ax2.axhline(1, linestyle='--', color='red', linewidth=1)
    ax2.set_title('Cumulative Returns: Strategy vs Benchmark')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Returns')
    ax2.legend()
    
    # Plot 3: Drawdown
    strategy_drawdown = calculate_drawdown(df_strategy[strategy_returns_column])
    benchmark_drawdown = calculate_drawdown(df_benchmark[benchmark_returns_column])
    
    ax3.plot(df_strategy['datetime'], strategy_drawdown, label='Strategy')
    ax3.plot(df_benchmark['datetime'], benchmark_drawdown, label='Benchmark')
    ax3.set_title('Drawdown: Strategy vs Benchmark')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def calculate_drawdown(returns):
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    return (wealth_index - previous_peaks) / previous_peaks