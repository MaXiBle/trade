"""
Основной модуль системы управления инвестиционным портфелем
Реализует основную логику мониторинга портфеля, ребалансировки и управления рисками
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import json
import yaml
from .data_loader import DataLoader
from .rebalancing_engine import RebalancingEngine
from .risk_manager import RiskManager


class PortfolioManager:
    """
    Основной класс для управления инвестиционным портфелем
    """
    
    def __init__(self, config_file: str = 'config/default_config.json'):
        """
        Инициализация менеджера портфеля с конфигурацией
        
        Args:
            config_file: Путь к файлу конфигурации
        """
        self.config = self._load_config(config_file)
        self.data_loader = DataLoader()
        self.rebalancing_engine = RebalancingEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Состояние портфеля
        self.portfolio_weights = {}
        self.portfolio_value = 0.0
        self.last_rebalance_date = None
        self.transaction_costs = 0.0
        
        # Инициализация портфеля
        self._initialize_portfolio()
    
    def _load_config(self, config_file: str) -> dict:
        """Загрузка конфигурации из файла"""
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                return json.load(f)
            elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                return yaml.safe_load(f)
        raise ValueError(f"Неподдерживаемый формат файла конфигурации: {config_file}")
    
    def _initialize_portfolio(self):
        """Инициализация портфеля с целевыми весами"""
        assets = self.config['assets']
        target_weights = self.config.get('target_weights', {})
        
        # Если целевые веса не предоставлены, распределяем равномерно
        if not target_weights:
            equal_weight = 1.0 / len(assets)
            self.portfolio_weights = {asset: equal_weight for asset in assets}
        else:
            self.portfolio_weights = target_weights.copy()
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        Расчет текущих весов портфеля на основе рыночных цен
        
        Returns:
            Словарь с текущими весами активов
        """
        # Получение текущих цен для всех активов
        current_prices = self.data_loader.get_current_prices(list(self.portfolio_weights.keys()))
        
        # Расчет рыночных стоимостей
        market_values = {}
        total_value = 0.0
        
        for asset, weight in self.portfolio_weights.items():
            if asset in current_prices:
                market_values[asset] = current_prices[asset]
                total_value += current_prices[asset]
        
        # Расчет текущих весов
        current_weights = {}
        for asset, value in market_values.items():
            current_weights[asset] = value / total_value if total_value > 0 else 0.0
        
        return current_weights
    
    def should_rebalance(self) -> bool:
        """
        Проверка необходимости ребалансировки портфеля на основе порога отклонения
        
        Returns:
            True если необходима ребалансировка, иначе False
        """
        current_weights = self.get_current_weights()
        target_weights = self.portfolio_weights
        
        # Проверка прошедшего минимального интервала ребалансировки
        min_interval = self.config.get('min_rebalance_interval_days', 7)
        if self.last_rebalance_date:
            days_since_rebalance = (datetime.now() - self.last_rebalance_date).days
            if days_since_rebalance < min_interval:
                return False
        
        # Проверка отклонения любого актива за пределы порога
        deviation_threshold = self.config.get('rebalance_threshold', 0.05)  # 5%
        
        for asset in target_weights:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights[asset]
            
            if abs(current_weight - target_weight) > deviation_threshold:
                return True
        
        return False
    
    def execute_rebalance(self):
        """
        Выполнение ребалансировки портфеля на основе текущих рыночных условий
        """
        if not self.should_rebalance():
            return
        
        # Расчет сигналов ребалансировки
        signals = self.rebalancing_engine.calculate_rebalance_signals(
            self.portfolio_weights,
            self.get_current_weights()
        )
        
        # Применение фильтров управления рисками
        filtered_signals = self.risk_manager.apply_filters(signals)
        
        # Выполнение ребалансировки
        self._apply_rebalance_signals(filtered_signals)
        
        # Обновление даты последней ребалансировки
        self.last_rebalance_date = datetime.now()
    
    def _apply_rebalance_signals(self, signals: Dict[str, float]):
        """
        Применение сигналов ребалансировки для обновления весов портфеля
        
        Args:
            signals: Словарь с сигналами ребалансировки для каждого актива
        """
        for asset, signal in signals.items():
            if asset in self.portfolio_weights:
                # Обновление веса портфеля на основе сигнала
                new_weight = self.portfolio_weights[asset] + signal
                # Обеспечение соблюдения границ веса
                max_weight = self.config.get('max_asset_weight', 0.20)  # 20%
                min_weight = self.config.get('min_asset_weight', 0.01)  # 1%
                
                self.portfolio_weights[asset] = max(min_weight, min(max_weight, new_weight))
    
    def get_portfolio_performance(self) -> Dict:
        """
        Расчет метрик производительности портфеля
        
        Returns:
            Словарь с метриками производительности
        """
        current_weights = self.get_current_weights()
        
        # Расчет общей стоимости портфеля
        current_prices = self.data_loader.get_current_prices(list(current_weights.keys()))
        total_value = sum(current_prices.values())
        
        return {
            'total_value': total_value,
            'current_weights': current_weights,
            'last_rebalance_date': self.last_rebalance_date,
            'transaction_costs': self.transaction_costs
        }
    
    def run(self):
        """
        Основной цикл выполнения менеджера портфеля
        """
        print("Запуск системы управления портфелем...")
        
        while True:
            try:
                # Мониторинг портфеля
                current_weights = self.get_current_weights()
                print(f"Текущие веса портфеля: {current_weights}")
                
                # Проверка необходимости ребалансировки
                if self.should_rebalance():
                    print("Обнаружен сигнал ребалансировки. Выполняется ребалансировка...")
                    self.execute_rebalance()
                    print(f"Веса портфеля после ребалансировки: {self.portfolio_weights}")
                else:
                    print("В данный момент ребалансировка не требуется.")
                
                # Ожидание перед следующей проверкой (в реальной системе это будет запланировано)
                break  # Для демонстрации выполним один раз
                
            except Exception as e:
                print(f"Ошибка в управлении портфелем: {e}")
                break
    
    def get_historical_performance(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Получение исторических данных производительности для бэктестирования
        
        Args:
            start_date: Начальная дата в формате ГГГГ-ММ-ДД
            end_date: Конечная дата в формате ГГГГ-ММ-ДД
            
        Returns:
            DataFrame с историческими данными производительности
        """
        return self.data_loader.get_historical_data(
            list(self.portfolio_weights.keys()),
            start_date,
            end_date
        )