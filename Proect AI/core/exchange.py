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
            
            # Выводим детальную информацию о полученном стакане
            asks_count = len(orderbook.get('asks', []))
            bids_count = len(orderbook.get('bids', []))
            timestamp = orderbook.get('timestamp')
            
            # Преобразуем timestamp в читаемый формат, если он есть
            time_str = "N/A"
            if timestamp:
                time_str = time.strftime('%H:%M:%S', time.localtime(timestamp / 1000))
                
            self.logger.info(f"REST API: Получен стакан {symbol} на {time_str} | {asks_count} заявок на продажу, {bids_count} заявок на покупку")
            
            # Показываем лучшие 3 цены спроса и предложения
            if asks_count > 0 and bids_count > 0:
                asks = orderbook.get('asks', [])[:3]
                bids = orderbook.get('bids', [])[:3]
                asks_str = " | ".join([f"{a[0]}:{a[1]}" for a in asks])
                bids_str = " | ".join([f"{b[0]}:{b[1]}" for b in bids])
                self.logger.info(f"REST API: Лучшие ASK: {asks_str}")
                self.logger.info(f"REST API: Лучшие BID: {bids_str}")
                
                # Расчет спреда
                best_ask = float(asks[0][0])
                best_bid = float(bids[0][0])
                spread = best_ask - best_bid
                spread_percent = (spread / best_bid) * 100
                self.logger.info(f"REST API: Спред: {spread:.2f} USD ({spread_percent:.4f}%)")
                
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
            
            # Выводим информацию о балансе
            total = balance.get('total', {})
            free = balance.get('free', {})
            used = balance.get('used', {})
            
            # Находим и выводим основные валюты
            main_currencies = ['USDT', 'BTC', 'ETH']
            for currency in main_currencies:
                if currency in total:
                    total_amount = total.get(currency, 0)
                    free_amount = free.get(currency, 0)
                    used_amount = used.get(currency, 0)
                    
                    if total_amount > 0:
                        self.logger.info(f"REST API: Баланс {currency}: {total_amount} (доступно: {free_amount}, используется: {used_amount})")
            
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
            
            # Выводим информацию о позициях
            positions_count = len(positions)
            self.logger.info(f"REST API: Получено {positions_count} позиций")
            
            # Показываем детали по каждой позиции
            for position in positions:
                symbol = position.get('symbol', 'UNKNOWN')
                side = position.get('side', 'UNKNOWN')
                contracts = position.get('contracts', 0)
                contract_size = position.get('contractSize', 1)
                notional = position.get('notional', 0)
                leverage = position.get('leverage', 1)
                
                # Определяем цвет для стороны
                side_color = Colors.GREEN if side == 'long' else Colors.RED
                side_text = f"{side_color}{side.upper()}{Colors.RESET}"
                
                # Проверяем, есть ли реальная позиция
                if notional and float(notional) != 0:
                    entry_price = position.get('entryPrice', 0)
                    unrealized_pnl = position.get('unrealizedPnl', 0)
                    
                    # Определяем цвет для PnL
                    if unrealized_pnl:
                        try:
                            pnl_float = float(unrealized_pnl)
                            if pnl_float > 0:
                                pnl_colored = f"{Colors.GREEN}+{unrealized_pnl}{Colors.RESET}"
                            elif pnl_float < 0:
                                pnl_colored = f"{Colors.RED}{unrealized_pnl}{Colors.RESET}"
                            else:
                                pnl_colored = str(unrealized_pnl)
                        except:
                            pnl_colored = str(unrealized_pnl)
                    else:
                        pnl_colored = "N/A"
                    
                    self.logger.info(f"REST API: Позиция {symbol}: {side_text} | Размер: {contracts} контрактов | Цена входа: {entry_price} | PnL: {pnl_colored} | Плечо: {leverage}x")
            
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
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ REST API СОЕДИНЕНИЯ")
    print("Вы увидите все данные, получаемые через REST API")
    print("=" * 50)
    
    # Создаем экземпляр биржи
    exchange = OKXExchange(config_path='../config.yaml')
    
    # Выводим информацию о текущем режиме
    print(f"Текущий режим: {exchange.mode}")
    print("-" * 50)
    
    # Запрашиваем информацию о тикере
    symbol = exchange.symbol  # Берем символ из конфигурации
    print(f"Получение информации для символа: {symbol}")
    print("-" * 50)
    
    print("1. Получаем информацию о тикере...")
    try:
        ticker = await exchange.fetch_ticker(symbol)
        print(f"Тикер для {symbol}:")
        print(f"  Последняя цена: {ticker['last']}")
        print(f"  Лучшая цена покупки: {ticker['bid']}")
        print(f"  Лучшая цена продажи: {ticker['ask']}")
        print(f"  24ч объем: {ticker['volume']} / {ticker['quoteVolume']} USDT")
        print(f"  24ч изменение: {ticker['percentage']}%")
        print(f"  24ч максимум: {ticker['high']}")
        print(f"  24ч минимум: {ticker['low']}")
        print(f"  Временная метка: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ticker['timestamp'] / 1000))}")
        print("-" * 50)
    except Exception as e:
        print(f"Ошибка при получении тикера: {e}")
    
    # Получаем стакан ордеров
    print("2. Получаем стакан ордеров (первые 5 уровней)...")
    try:
        orderbook = await exchange.fetch_orderbook(symbol, 5)
        # Информация уже выводится внутри метода fetch_orderbook
        print("-" * 50)
    except Exception as e:
        print(f"Ошибка при получении стакана ордеров: {e}")
    
    # Получаем баланс (если аутентифицированы)
    print("3. Получаем баланс аккаунта...")
    try:
        balance = await exchange.fetch_balance()
        # Информация уже выводится внутри метода fetch_balance
        print("-" * 50)
    except Exception as e:
        print(f"Ошибка при получении баланса: {e}")
    
    # Получаем открытые позиции
    print("4. Получаем открытые позиции...")
    try:
        positions = await exchange.fetch_positions(symbol)
        # Информация уже выводится внутри метода fetch_positions
        print("-" * 50)
    except Exception as e:
        print(f"Ошибка при получении позиций: {e}")
    
    print("Тестирование REST API завершено")
    print("=" * 50)

if __name__ == "__main__":
    # Запускаем тест с детальным выводом данных
    try:
        asyncio.run(test_exchange())
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем")
    except Exception as e:
        print(f"Ошибка при выполнении теста: {e}") 