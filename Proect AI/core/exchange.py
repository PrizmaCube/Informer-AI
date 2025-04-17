import os
import sys
import yaml
import ccxt
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import time

# Добавляем родительскую директорию в путь для импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ANSI коды цветов для терминала
class Colors:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

# Форматтер с поддержкой цветов
class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BLUE,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD
    }

    def format(self, record):
        # Добавление цвета в зависимости от уровня логирования
        levelname = record.levelname
        if record.levelno in self.LEVEL_COLORS:
            colored_levelname = self.LEVEL_COLORS[record.levelno] + levelname + Colors.RESET
            record.levelname = colored_levelname
        return super().format(record)

class OKXExchange:
    """
    Класс для работы с биржей OKX через библиотеку CCXT.
    Поддерживает переключение между тестовым (demo) и боевым (live) режимами.
    """
    
    def __init__(self, config_path: str = '../config.yaml'):
        """
        Инициализация клиента OKX
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        self.mode = self.config.get('mode', 'demo')
        self.exchange = None
        self.ws_connection = None
        self.symbol = self.config.get('trading', {}).get('symbol', 'ETH-USDT-SWAP')
        
        # Инициализация REST API клиента
        self._initialize_exchange()
        
        self.logger.info(f"Клиент OKX инициализирован в режиме {self.mode}")
        
    def _setup_logger(self):
        """Настройка логгера с цветным форматированием"""
        logger = logging.getLogger('okx_exchange')
        logger.setLevel(logging.INFO)
        
        # Проверяем, есть ли уже обработчики у логгера
        if not logger.handlers:
            # Обработчик для вывода в консоль с цветным форматированием
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Формат логов с цветами для консоли
            colored_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(colored_formatter)
            
            # Добавляем обработчик к логгеру
            logger.addHandler(console_handler)
        
        return logger
        
    def _load_config(self, config_path: str) -> dict:
        """
        Загрузка конфигурации из YAML файла
        
        Args:
            config_path: Путь к файлу конфигурации
            
        Returns:
            dict: Загруженная конфигурация
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {e}")
            raise
            
    def _initialize_exchange(self):
        """Инициализация объекта биржи с настройками API"""
        try:
            # Проверяем наличие необходимых API ключей
            if not self.config.get('okx', {}).get(self.mode, {}).get('api_key') or not self.config.get('okx', {}).get(self.mode, {}).get('api_secret') or not self.config.get('okx', {}).get(self.mode, {}).get('password'):
                self.logger.error(f"Невозможно инициализировать биржу: отсутствуют API ключи для режима {self.mode}")
                raise ValueError(f"API ключи не указаны в конфигурации для режима {self.mode}")
            
            # Создаем объект биржи OKX
            self.exchange = ccxt.okex({
                'apiKey': self.config.get('okx', {}).get(self.mode, {}).get('api_key'),
                'secret': self.config.get('okx', {}).get(self.mode, {}).get('api_secret'),
                'password': self.config.get('okx', {}).get(self.mode, {}).get('password'),
                'timeout': 30000,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                }
            })
            
            # Настройка для тестового режима
            if self.mode == 'demo':
                self.exchange.options['testnet'] = True
                
                # Правильная настройка URL для демо-режима (без дублирования api/v5)
                self.exchange.urls = {
                    'api': {
                        'rest': 'https://www.okx.com',
                        'ws': 'wss://wspap.okx.com:8443/ws/v5'
                    }
                }
                
                self.logger.info("Биржа инициализирована в ДЕМО режиме")
            else:
                self.logger.info("Биржа инициализирована в БОЕВОМ режиме")
                
            # Устанавливаем параметры запросов    
            self.exchange.options['recvWindow'] = 10000
            
            # Проверка соединения с API
            self.logger.info("Проверка соединения с API OKX...")
            time_response = self.exchange.fetch_time()
            self.logger.info(f"Соединение с API OKX установлено успешно. Время сервера: {time_response}")
            
            return self.exchange
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации биржи: {e}")
            raise
    
    def switch_mode(self, new_mode: str):
        """
        Переключение между тестовым и боевым режимами
        
        Args:
            new_mode: Новый режим ('demo' или 'live')
        """
        if new_mode not in ['demo', 'live']:
            raise ValueError("Режим должен быть 'demo' или 'live'")
            
        if new_mode == self.mode:
            self.logger.info(f"Уже в режиме {new_mode}")
            return
            
        self.logger.info(f"Переключение с режима {self.mode} на {new_mode}")
        self.mode = new_mode
        
        # Переинициализируем обмен с новыми параметрами
        self._initialize_exchange()
        
        self.logger.info(f"Успешно переключено на режим {new_mode}")
    
    # Методы для работы с REST API
    
    async def fetch_ticker(self, symbol: Optional[str] = None) -> dict:
        """
        Получение текущей информации о тикере
        
        Args:
            symbol: Символ торговой пары (если None, используется self.symbol)
            
        Returns:
            dict: Информация о тикере
        """
        symbol = symbol or self.symbol
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            self.logger.error(f"Ошибка получения тикера для {symbol}: {e}")
            raise
    
    async def fetch_orderbook(self, symbol: Optional[str] = None, limit: int = 400) -> dict:
        """
        Получение стакана ордеров
        
        Args:
            symbol: Символ торговой пары (если None, используется self.symbol)
            limit: Глубина стакана (до 400 уровней)
            
        Returns:
            dict: Стакан ордеров
        """
        symbol = symbol or self.symbol
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            self.logger.error(f"Ошибка получения стакана ордеров для {symbol}: {e}")
            raise
    
    async def fetch_balance(self) -> dict:
        """
        Получение баланса аккаунта
        
        Returns:
            dict: Информация о балансе
        """
        try:
            balance = await self.exchange.fetch_balance()
            return balance
        except Exception as e:
            self.logger.error(f"Ошибка получения баланса: {e}")
            raise
            
    async def fetch_positions(self, symbol: Optional[str] = None) -> List[dict]:
        """
        Получение открытых позиций
        
        Args:
            symbol: Символ торговой пары (если None, используются все позиции)
            
        Returns:
            List[dict]: Список открытых позиций
        """
        try:
            positions = await self.exchange.fetch_positions(symbol)
            return positions
        except Exception as e:
            self.logger.error(f"Ошибка получения позиций: {e}")
            raise
            
    # Методы для торговли
            
    async def create_order(self, side: str, amount: float, 
                           price: Optional[float] = None, 
                           order_type: str = 'limit',
                           params: dict = {}) -> dict:
        """
        Создание ордера
        
        Args:
            side: Сторона ('buy' или 'sell')
            amount: Объем
            price: Цена (для лимитных ордеров)
            order_type: Тип ордера ('limit' или 'market')
            params: Дополнительные параметры
            
        Returns:
            dict: Результат создания ордера
        """
        try:
            order = await self.exchange.create_order(
                symbol=self.symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            self.logger.info(f"Создан {order_type} ордер {side} на {amount} {self.symbol} по цене {price}")
            return order
        except Exception as e:
            self.logger.error(f"Ошибка создания ордера: {e}")
            raise

    # Методы для работы с WebSocket API будут добавлены позже
    # Они будут включать подписку на обновления стакана, сделок и т.д.

# Функция для тестирования модуля
async def test_exchange():
    """Тестирование функциональности обмена"""
    # Создаем экземпляр биржи
    exchange = OKXExchange(config_path='../config.yaml')
    
    # Выводим информацию о текущем режиме
    print(f"Текущий режим: {exchange.mode}")
    
    # Получаем текущую информацию о тикере
    ticker = await exchange.fetch_ticker()
    print(f"Тикер: {ticker['last']}")
    
    # Получаем стакан ордеров (первые 5 уровней)
    orderbook = await exchange.fetch_orderbook(limit=5)
    print(f"Топ 5 ордеров на покупку: {orderbook['bids']}")
    print(f"Топ 5 ордеров на продажу: {orderbook['asks']}")
    
    # Получаем баланс
    balance = await exchange.fetch_balance()
    print(f"Баланс USDT: {balance.get('USDT', {}).get('free', 0)}")
    
    # Переключаемся на другой режим
    new_mode = 'live' if exchange.mode == 'demo' else 'demo'
    exchange.switch_mode(new_mode)
    print(f"Переключено на режим {exchange.mode}")
    
    # Снова получаем информацию о тикере
    ticker = await exchange.fetch_ticker()
    print(f"Тикер в режиме {exchange.mode}: {ticker['last']}")

if __name__ == "__main__":
    # Запускаем тест
    asyncio.run(test_exchange()) 