#!/usr/bin/env python3
"""
Модуль тестирования для симуляции рынка на исторических данных
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtester import Backtester
from src.data_loader import DataLoader
import matplotlib.pyplot as plt


def simulate_market_over_interval(symbols, start_date, end_date):
    """
    Симуляция рынка за заданный исторический интервал
    
    Args:
        symbols: Список тикеров акций
        start_date: Начальная дата (ГГГГ-ММ-ДД)
        end_date: Конечная дата (ГГГГ-ММ-ДД)
        
    Returns:
        DataFrame с историческими данными цен
    """
    print(f"Симуляция рынка для {symbols} с {start_date} по {end_date}")
    
    # Загрузка исторических данных
    data_loader = DataLoader()
    price_data = data_loader.get_historical_data(symbols, start_date, end_date)
    
    if price_data.empty:
        print("Нет исторических данных для симуляции")
        return pd.DataFrame()
    
    # Расчет доходностей
    returns_data = price_data.pct_change().dropna()
    
    print(f"Данные охвата: {len(price_data)} торговых дней")
    print(f"Диапазон дат: {price_data.index[0]} до {price_data.index[-1]}")
    print("\nСтатистика по активам:")
    
    for symbol in symbols:
        if symbol in price_data.columns:
            prices = price_data[symbol].dropna()
            returns = returns_data[symbol].dropna()
            
            print(f"\n{symbol}:")
            print(f"  Начальная цена: ${prices.iloc[0]:.2f}")
            print(f"  Конечная цена: ${prices.iloc[-1]:.2f}")
            print(f"  Общая доходность: {(prices.iloc[-1]/prices.iloc[0] - 1):.2%}")
            print(f"  Средняя доходность: {returns.mean():.2%}")
            print(f"  Волатильность: {returns.std():.2%}")
            print(f"  Максимальная просадка: {calculate_max_drawdown(prices):.2%}")
    
    return price_data


def calculate_max_drawdown(price_series):
    """Расчет максимальной просадки"""
    rolling_max = price_series.expanding().max()
    drawdown = (price_series - rolling_max) / rolling_max
    return drawdown.min()


def run_portfolio_simulation_test():
    """
    Запуск теста симуляции портфеля на исторических данных
    """
    print("="*70)
    print("ТЕСТИРОВАНИЕ СИМУЛЯЦИИ РЫНКА НА ИСТОРИЧЕСКИХ ДАННЫХ")
    print("="*70)
    
    # Определение конфигурации для теста
    config = {
        'rebalance_threshold': 0.05,      # Ребалансировка при отклонении веса более чем на 5%
        'min_rebalance_interval_days': 7, # Минимум 7 дней между ребалансировками
        'max_weight_limit': 0.25,         # Максимум 25% в любом активе
        'critical_drop_threshold': -0.30, # Не инвестировать в активы, упавшие более чем на 30% за месяц
        'transaction_cost': 0.001,        # Комиссия 0.1% за транзакцию
        'initial_weights': {              # Начальные веса для портфеля
            'AAPL': 0.2,
            'MSFT': 0.2,
            'GOOGL': 0.2,
            'AMZN': 0.2,
            'TSLA': 0.2
        }
    }
    
    # Тестовые символы и временной период
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    
    # Симуляция рынка
    market_data = simulate_market_over_interval(symbols, start_date, end_date)
    
    if market_data.empty:
        print("Невозможно провести тест из-за отсутствия данных")
        return
    
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ СТРАТЕГИИ РЕБАЛАНСИРОВКИ НА ИСТОРИЧЕСКИХ ДАННЫХ")
    print("="*70)
    
    # Создание экземпляра бэктестера
    backtester = Backtester(config)
    
    try:
        # Запуск бэктеста
        print(f"Запуск симуляции портфельной стратегии с {start_date} по {end_date}...")
        results = backtester.run_backtest(symbols, start_date, end_date)
        
        # Отображение результатов
        print("\nРЕЗУЛЬТАТЫ СИМУЛЯЦИИ:")
        print("-" * 50)
        
        # Общее изменение баланса портфеля
        print(f"Изменение баланса портфеля (с ребалансировкой): {results['total_return']:.2%}")
        print(f"Изменение баланса портфеля (бенчмарк):        {results['benchmark_total_return']:.2%}")
        print(f"Избыточная доходность:                         {results['total_return'] - results['benchmark_total_return']:.2%}")
        print()
        
        print("ПОЯСНЕНИЕ МЕТРИК:")
        print("-" * 50)
        print("Общая доходность (Total Return): Общее изменение стоимости портфеля за весь период")
        print("Годовая доходность (Annualized Return): Среднегодовая доходность, рассчитанная с учетом сложного процента")
        print("Волатильность (Volatility): Стандартное отклонение ежедневных доходностей, мера риска")
        print("Коэффициент Шарпа (Sharpe Ratio): Показатель эффективности, соотношение доходности к риску")
        print("Максимальная просадка (Max Drawdown): Максимальное падение стоимости портфеля от предыдущего максимума")
        print("Альфа (Alpha): Избыточная доходность портфеля по сравнению с бенчмарком")
        print("Бета (Beta): Мера чувствительности портфеля к движению рынка")
        print("Информационное соотношение (Information Ratio): Отношение избыточной доходности к отслеживанию ошибки")
        print("Количество ребалансировок: Сколько раз происходила ребалансировка портфеля")
        print()
        
        print(f"Годовая доходность (с ребалансировкой):  {results['annualized_return']:.2%}")
        print(f"Годовая доходность (бенчмарк):           {results['benchmark_annualized_return']:.2%}")
        print()
        
        print(f"Волатильность (с ребалансировкой):        {results['volatility']:.2%}")
        print(f"Волатильность (бенчмарк):                {results['benchmark_volatility']:.2%}")
        print()
        
        print(f"Коэффициент Шарпа (с ребалансировкой):    {results['sharpe_ratio']:.3f}")
        print(f"Коэффициент Шарпа (бенчмарк):            {results['benchmark_sharpe']:.3f}")
        print()
        
        print(f"Максимальная просадка (с ребалансировкой): {results['max_drawdown']:.2%}")
        print(f"Максимальная просадка (бенчмарк):          {results['benchmark_max_drawdown']:.2%}")
        print()
        
        print(f"Альфа:                                     {results['alpha']:.3f}")
        print(f"Бета:                                      {results['beta']:.3f}")
        print(f"Информационное соотношение:                {results['information_ratio']:.3f}")
        print()
        
        print(f"Количество ребалансировок:                 {results['num_rebalances']}")
        
        # Генерация подробного отчета
        print("\nПОДРОБНЫЙ ОТЧЕТ:")
        print("=" * 70)
        report = backtester.generate_report()
        print(report)
        
        # Построение графиков (раскомментируйте, если хотите визуализировать)
        # backtester.plot_results()
        
    except Exception as e:
        print(f"Ошибка во время симуляции: {str(e)}")
        import traceback
        traceback.print_exc()


def run_simple_simulation():
    """
    Запуск простой симуляции для быстрой проверки
    """
    print("\n" + "="*50)
    print("ПРОСТАЯ СИМУЛЯЦИЯ РЫНКА")
    print("="*50)
    
    # Простая конфигурация с равными весами
    config = {
        'rebalance_threshold': 0.05,
        'min_rebalance_interval_days': 7,
        'max_weight_limit': 0.30,
        'transaction_cost': 0.001
    }
    
    backtester = Backtester(config)
    
    # Тест с меньшим портфелем для более быстрого выполнения
    symbols = ['SPY', 'QQQ', 'TLT']  # ETF для диверсификации
    start_date = '2021-01-01'
    end_date = '2022-01-01'
    
    print(f"Симуляция: {', '.join(symbols)} с {start_date} по {end_date}")
    
    try:
        # Сначала симуляция рынка
        simulate_market_over_interval(symbols, start_date, end_date)
        
        # Затем тест стратегии
        results = backtester.run_backtest(symbols, start_date, end_date)
        
        print(f"\nРЕЗУЛЬТАТЫ СИМУЛЯЦИИ:")
        print(f"Доходность портфеля (с ребалансировкой): {results['total_return']:.2%}")
        print(f"Доходность бенчмарка:                    {results['benchmark_total_return']:.2%}")
        print(f"Избыточная доходность:                   {results['total_return'] - results['benchmark_total_return']:.2%}")
        print(f"Количество ребалансировок:               {results['num_rebalances']}")
        print(f"Максимальная просадка (портфель):        {results['max_drawdown']:.2%}")
        print(f"Максимальная просадка (бенчмарк):        {results['benchmark_max_drawdown']:.2%}")
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")


if __name__ == "__main__":
    run_portfolio_simulation_test()
    run_simple_simulation()