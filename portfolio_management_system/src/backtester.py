"""
Модуль бэктестирования для системы управления портфелем
Тестирует стратегию ребалансировки на исторических данных
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


class Backtester:
    """
    Класс для бэктестирования стратегий портфеля на исторических данных
    """
    
    def __init__(self, config: dict):
        """
        Инициализация бэктестера с конфигурацией
        
        Args:
            config: Словарь конфигурации
        """
        self.config = config
        self.results = {}
        self.portfolio_history = []
        self.benchmark_history = []
        self.rebalance_history = []
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """
        Запуск бэктеста стратегии портфеля
        
        Args:
            symbols: Список тикеров акций для включения в портфель
            start_date: Начальная дата бэктеста (ГГГГ-ММ-ДД)
            end_date: Конечная дата бэктеста (ГГГГ-ММ-ДД)
            
        Returns:
            Словарь с результатами бэктеста
        """
        print(f"Запуск бэктеста с {start_date} по {end_date}")
        
        # Загрузка исторических данных
        from .data_loader import DataLoader
        data_loader = DataLoader()
        price_data = data_loader.get_historical_data(symbols, start_date, end_date)
        
        if price_data.empty:
            raise ValueError("Нет исторических данных для бэктестирования")
        
        # Расчет ежедневной доходности
        returns_data = price_data.pct_change().dropna()
        
        # Инициализация начальных весов портфеля (равные веса или из конфига)
        if 'initial_weights' in self.config:
            initial_weights = self.config['initial_weights']
        else:
            n_assets = len(symbols)
            initial_weights = {symbol: 1.0/n_assets for symbol in symbols}
        
        # Инициализация параметров риска
        max_weight_limit = self.config.get('max_weight_limit', 0.20)  # максимум 20% на актив
        critical_drop_threshold = self.config.get('critical_drop_threshold', -0.30)  # порог -30%
        rebalance_threshold = self.config.get('rebalance_threshold', 0.05)  # порог 5%
        min_rebalance_interval = self.config.get('min_rebalance_interval_days', 7)  # 7 дней
        transaction_cost = self.config.get('transaction_cost', 0.001)  # 0.1% за транзакцию
        
        # Симуляция производительности портфеля во времени
        portfolio_values = []
        portfolio_weights_history = []
        rebalance_dates = []
        
        # Начальное значение
        current_weights = initial_weights.copy()
        initial_value = 100000  # начальный капитал $100k
        current_value = initial_value
        last_rebalance_date = None
        
        # Итерация по каждому дню
        for date_idx in range(len(returns_data)):
            current_date = returns_data.index[date_idx]
            
            # Расчет доходности портфеля за этот день на основе вчерашних весов
            day_return = 0
            for symbol in symbols:
                if symbol in returns_data.columns:
                    day_return += current_weights.get(symbol, 0) * returns_data.iloc[date_idx][symbol]
            
            # Обновление стоимости портфеля
            current_value *= (1 + day_return)
            portfolio_values.append({
                'date': current_date,
                'value': current_value,
                'return': day_return
            })
            
            # Сохранение истории портфеля
            self.portfolio_history.append({
                'date': current_date,
                'value': current_value,
                'return': day_return,
                'weights': current_weights.copy()
            })
            
            # Расчет стоимости бенчмарка (покупка и удержание без ребалансировки)
            if date_idx == 0:
                benchmark_value = initial_value
            else:
                benchmark_return = returns_data.iloc[date_idx].mean()  # равновзвешенный бенчмарк
                benchmark_value *= (1 + benchmark_return)
            
            self.benchmark_history.append({
                'date': current_date,
                'value': benchmark_value
            })
            
            # Проверка необходимости ребалансировки (после первого дня)
            if date_idx > 0:
                # Расчет текущих весов на основе производительности
                current_prices = price_data.loc[current_date]
                prev_prices = price_data.iloc[date_idx-1]
                
                # Расчет текущих рыночных стоимостей и весов
                market_values = {}
                for symbol in symbols:
                    if symbol in current_prices and symbol in prev_prices:
                        # Расчет стоимости актива на основе стоимости портфеля и доходности
                        prev_weight = current_weights.get(symbol, initial_weights.get(symbol, 0))
                        price_return = (current_prices[symbol] / prev_prices[symbol]) - 1
                        new_value = current_value * prev_weight * (1 + price_return)
                        market_values[symbol] = new_value
                
                total_value = sum(market_values.values())
                if total_value > 0:
                    actual_weights = {s: v/total_value for s, v in market_values.items()}
                else:
                    actual_weights = current_weights.copy()
                
                # Проверка, упал ли какой-либо актив критически (более чем на -30% за месяц)
                should_skip_rebalance = False
                if date_idx >= 22:  # По крайней мере месяц данных
                    for symbol in symbols:
                        if symbol in price_data.columns:
                            # Расчет месячной доходности
                            month_ago_idx = date_idx - 22  # Приблизительно месяц
                            if month_ago_idx >= 0:
                                month_return = (current_prices[symbol] / price_data.iloc[month_ago_idx][symbol]) - 1
                                if month_return < critical_drop_threshold:
                                    print(f"Обнаружено критическое падение для {symbol}: {month_return:.2%}, пропускаем ребалансировку")
                                    should_skip_rebalance = True
                                    break
                
                # Проверка условий ребалансировки
                days_since_last_rebalance = (
                    (current_date - last_rebalance_date).days 
                    if last_rebalance_date else float('inf')
                )
                
                weight_drift_exceeded = any(
                    abs(actual_weights.get(s, 0) - initial_weights.get(s, 0)) > rebalance_threshold 
                    for s in symbols
                )
                
                should_rebalance = (
                    not should_skip_rebalance and
                    days_since_last_rebalance >= min_rebalance_interval and
                    weight_drift_exceeded
                )
                
                if should_rebalance:
                    # Применение ограничений риска к целевым весам
                    adjusted_weights = self._apply_risk_constraints(actual_weights, initial_weights, max_weight_limit)
                    
                    # Расчет транзакционных издержек
                    weight_changes = {s: abs(adjusted_weights.get(s, 0) - current_weights.get(s, 0)) 
                                      for s in symbols}
                    total_transaction_cost = sum(weight_changes.values()) * transaction_cost
                    
                    # Применение транзакционных издержек к стоимости портфеля
                    current_value *= (1 - total_transaction_cost)
                    
                    # Обновление весов после ребалансировки
                    current_weights = adjusted_weights
                    last_rebalance_date = current_date
                    rebalance_dates.append(current_date)
                    
                    # Сохранение события ребалансировки
                    self.rebalance_history.append({
                        'date': current_date,
                        'previous_weights': actual_weights.copy(),
                        'new_weights': current_weights.copy(),
                        'transaction_cost': total_transaction_cost
                    })
            
            portfolio_weights_history.append(current_weights.copy())
        
        # Преобразование в DataFrame
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        benchmark_df = pd.DataFrame(self.benchmark_history)
        benchmark_df.set_index('date', inplace=True)
        
        # Расчет метрик производительности
        self.results = self._calculate_metrics(portfolio_df, benchmark_df, rebalance_dates)
        
        return self.results
    
    def _apply_risk_constraints(self, current_weights: Dict[str, float], target_weights: Dict[str, float], 
                               max_weight_limit: float) -> Dict[str, float]:
        """
        Применение ограничений управления рисками к весам ребалансировки
        
        Args:
            current_weights: Текущие веса портфеля
            target_weights: Целевые веса для ребалансировки
            max_weight_limit: Максимально допустимый вес для любого актива
            
        Returns:
            Скорректированные веса, соответствующие ограничениям риска
        """
        adjusted_weights = target_weights.copy()
        
        # Обеспечение того, что ни один актив не превышает максимальный лимит веса
        for symbol, weight in adjusted_weights.items():
            if weight > max_weight_limit:
                adjusted_weights[symbol] = max_weight_limit
        
        # Нормализация весов, если они превышают 100%
        total_weight = sum(adjusted_weights.values())
        if total_weight > 1:
            for symbol in adjusted_weights:
                adjusted_weights[symbol] /= total_weight
        
        # Обеспечение всех весов положительными и суммирующимися в 1
        for symbol in adjusted_weights:
            if adjusted_weights[symbol] < 0:
                adjusted_weights[symbol] = 0
        
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for symbol in adjusted_weights:
                adjusted_weights[symbol] /= total_weight
        
        return adjusted_weights
    
    def _calculate_metrics(self, portfolio_data: pd.DataFrame, benchmark_data: pd.DataFrame, 
                          rebalance_dates: List) -> Dict:
        """
        Расчет метрик производительности из результатов бэктеста
        
        Args:
            portfolio_data: DataFrame с историей стоимости портфеля
            benchmark_data: DataFrame с историей стоимости бенчмарка
            rebalance_dates: Список дат, когда происходила ребалансировка
            
        Returns:
            Словарь с метриками производительности
        """
        # Доходность портфеля
        portfolio_returns = portfolio_data['value'].pct_change().dropna()
        benchmark_returns = benchmark_data['value'].pct_change().dropna()
        
        # Общая доходность
        total_return = (portfolio_data['value'].iloc[-1] / portfolio_data['value'].iloc[0]) - 1
        benchmark_total_return = (benchmark_data['value'].iloc[-1] / benchmark_data['value'].iloc[0]) - 1
        
        # Годовая доходность (предполагаем 252 торговых дня в году)
        years = len(portfolio_data) / 252
        annualized_return = (portfolio_data['value'].iloc[-1] / portfolio_data['value'].iloc[0]) ** (1/years) - 1 if years > 0 else 0
        benchmark_annualized_return = (benchmark_data['value'].iloc[-1] / benchmark_data['value'].iloc[0]) ** (1/years) - 1 if years > 0 else 0
        
        # Волатильность (годовая)
        volatility = portfolio_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        
        # Максимальная просадка
        rolling_max = portfolio_data['value'].expanding().max()
        daily_drawdown = portfolio_data['value'] / rolling_max - 1
        max_drawdown = daily_drawdown.min()
        
        benchmark_rolling_max = benchmark_data['value'].expanding().max()
        benchmark_daily_drawdown = benchmark_data['value'] / benchmark_rolling_max - 1
        benchmark_max_drawdown = benchmark_daily_drawdown.min()
        
        # Коэффициент Шарпа (предполагаем нулевую безрисковую ставку для простоты)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        benchmark_sharpe = benchmark_annualized_return / benchmark_volatility if benchmark_volatility != 0 else 0
        
        # Количество событий ребалансировки
        num_rebalances = len(rebalance_dates)
        
        # Альфа и Бета расчет
        # Ковариация доходности портфеля и бенчмарка
        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # Альфа (избыточная доходность над тем, что предсказывает CAPM)
        alpha = annualized_return - (0 + beta * (benchmark_annualized_return - 0))  # Предполагаем 0% безрисковую ставку
        
        # Информационное соотношение
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (annualized_return - benchmark_annualized_return) / tracking_error if tracking_error != 0 else 0
        
        return {
            'total_return': total_return,
            'benchmark_total_return': benchmark_total_return,
            'annualized_return': annualized_return,
            'benchmark_annualized_return': benchmark_annualized_return,
            'volatility': volatility,
            'benchmark_volatility': benchmark_volatility,
            'sharpe_ratio': sharpe_ratio,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'num_rebalances': num_rebalances,
            'rebalance_dates': rebalance_dates
        }
    
    def plot_results(self, save_path: str = None):
        """
        Построение результатов бэктеста
        
        Args:
            save_path: Необязательный путь для сохранения графика
        """
        if not self.results or not self.portfolio_history:
            print("Нет результатов бэктеста для построения. Сначала запустите бэктест.")
            return
        
        # Преобразование истории в DataFrame для построения
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        benchmark_df = pd.DataFrame(self.benchmark_history)
        benchmark_df.set_index('date', inplace=True)
        
        # Создание подграфиков
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Результаты бэктестирования портфеля', fontsize=16)
        
        # График 1: Производительность портфеля и бенчмарка
        axes[0, 0].plot(portfolio_df.index, portfolio_df['value'], label='Портфель с ребалансировкой', linewidth=2)
        axes[0, 0].plot(benchmark_df.index, benchmark_df['value'], label='Бенчмарк покупки и удержания', linewidth=2)
        axes[0, 0].set_title('Стоимость портфеля со временем')
        axes[0, 0].set_xlabel('Дата')
        axes[0, 0].set_ylabel('Стоимость портфеля ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Добавление маркеров ребалансировки, если они есть
        if self.rebalance_history:
            rebalance_dates = [r['date'] for r in self.rebalance_history]
            rebalance_values = [portfolio_df.loc[r['date']]['value'] for r in self.rebalance_history]
            axes[0, 0].scatter(rebalance_dates, rebalance_values, color='red', marker='^', s=50, label='Точки ребалансировки')
        
        # График 2: График просадки
        portfolio_rolling_max = portfolio_df['value'].expanding().max()
        portfolio_drawdown = (portfolio_df['value'] - portfolio_rolling_max) / portfolio_rolling_max
        benchmark_rolling_max = benchmark_df['value'].expanding().max()
        benchmark_drawdown = (benchmark_df['value'] - benchmark_rolling_max) / benchmark_rolling_max
        
        axes[0, 1].fill_between(portfolio_df.index, portfolio_drawdown * 100, 0, alpha=0.3, color='blue', label='Просадка портфеля')
        axes[0, 1].fill_between(benchmark_df.index, benchmark_drawdown * 100, 0, alpha=0.3, color='orange', label='Просадка бенчмарка')
        axes[0, 1].set_title('Сравнение просадки')
        axes[0, 1].set_xlabel('Дата')
        axes[0, 1].set_ylabel('Просадка (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_yaxis()  # Инвертировать ось y, чтобы просадки отображались как отрицательные значения ниже нуля
        
        # График 3: Веса активов со временем (если есть история весов)
        if len(self.portfolio_history) > 0 and 'weights' in self.portfolio_history[0]:
            # Получение всех уникальных символов из весов
            all_symbols = set()
            for entry in self.portfolio_history:
                all_symbols.update(entry['weights'].keys())
            
            # Создание DataFrame для весов со временем
            weight_data = {}
            for symbol in all_symbols:
                weight_data[symbol] = []
            
            for entry in self.portfolio_history:
                for symbol in all_symbols:
                    weight_data[symbol].append(entry['weights'].get(symbol, 0))
            
            weight_df = pd.DataFrame(weight_data, index=portfolio_df.index)
            
            # Построение весов со временем
            for symbol in all_symbols:
                axes[1, 0].plot(weight_df.index, weight_df[symbol], label=symbol)
            axes[1, 0].set_title('Веса активов со временем')
            axes[1, 0].set_xlabel('Дата')
            axes[1, 0].set_ylabel('Вес')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # График 4: Сравнение метрик производительности
        metrics_labels = ['Общая доходность', 'Годовая доходность', 'Коэффициент Шарпа', 'Максимальная просадка']
        portfolio_metrics = [
            self.results['total_return'],
            self.results['annualized_return'],
            self.results['sharpe_ratio'],
            self.results['max_drawdown']
        ]
        benchmark_metrics = [
            self.results['benchmark_total_return'],
            self.results['benchmark_annualized_return'],
            self.results['benchmark_sharpe'],
            self.results['benchmark_max_drawdown']
        ]
        
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, portfolio_metrics, width, label='Портфель с ребалансировкой', alpha=0.8)
        axes[1, 1].bar(x + width/2, benchmark_metrics, width, label='Бенчмарк покупки и удержания', alpha=0.8)
        axes[1, 1].set_title('Сравнение метрик производительности')
        axes[1, 1].set_xlabel('Метрики')
        axes[1, 1].set_ylabel('Значение')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics_labels, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Генерация текстового отчета результатов бэктестирования
        
        Returns:
            Отформатированная строка с результатами бэктестирования
        """
        if not self.results:
            return "Нет результатов бэктеста. Сначала запустите бэктест."
        
        report = []
        report.append("="*60)
        report.append("ОТЧЕТ О БЭКТЕСТЕ ПОРТФЕЛЯ")
        report.append("="*60)
        
        report.append(f"Период бэктеста: {self.portfolio_history[0]['date'].strftime('%Y-%m-%d')} по {self.portfolio_history[-1]['date'].strftime('%Y-%m-%d')}")
        report.append(f"Количество торговых дней: {len(self.portfolio_history)}")
        report.append(f"Количество ребалансировок: {self.results['num_rebalances']}")
        report.append("")
        
        report.append("МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
        report.append("-" * 30)
        report.append(f"Общая доходность портфеля (с ребалансировкой):     {self.results['total_return']:.2%}")
        report.append(f"Общая доходность бенчмарка:                      {self.results['benchmark_total_return']:.2%}")
        report.append(f"Избыточная доходность:                           {self.results['total_return'] - self.results['benchmark_total_return']:.2%}")
        report.append("")
        report.append(f"Годовая доходность портфеля (с ребалансировкой):  {self.results['annualized_return']:.2%}")
        report.append(f"Годовая доходность бенчмарка:                    {self.results['benchmark_annualized_return']:.2%}")
        report.append("")
        report.append(f"Волатильность портфеля (с ребалансировкой):       {self.results['volatility']:.2%}")
        report.append(f"Волатильность бенчмарка:                         {self.results['benchmark_volatility']:.2%}")
        report.append("")
        report.append(f"Коэффициент Шарпа портфеля (с ребалансировкой):   {self.results['sharpe_ratio']:.3f}")
        report.append(f"Коэффициент Шарпа бенчмарка:                     {self.results['benchmark_sharpe']:.3f}")
        report.append("")
        report.append(f"Максимальная просадка портфеля (с ребалансировкой): {self.results['max_drawdown']:.2%}")
        report.append(f"Максимальная просадка бенчмарка:                  {self.results['benchmark_max_drawdown']:.2%}")
        report.append("")
        report.append(f"Альфа портфеля:                                   {self.results['alpha']:.3f}")
        report.append(f"Бета портфеля:                                    {self.results['beta']:.3f}")
        report.append(f"Информационное соотношение:                      {self.results['information_ratio']:.3f}")
        report.append("")
        
        if self.rebalance_history:
            report.append("СОБЫТИЯ РЕБАЛАНСИРОВКИ:")
            report.append("-" * 30)
            for i, event in enumerate(self.rebalance_history[:5]):  # Показать первые 5 событий
                report.append(f"Событие {i+1}: {event['date'].strftime('%Y-%m-%d')}, Транзакционные издержки: {event['transaction_cost']:.3%}")
            if len(self.rebalance_history) > 5:
                report.append(f"... и еще {len(self.rebalance_history) - 5} событий")
        
        report.append("")
        report.append("="*60)
        
        return "\n".join(report)