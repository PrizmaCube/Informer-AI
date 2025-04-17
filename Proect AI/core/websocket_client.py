import os
import sys
import json
import yaml
import time
import hmac
import base64
import asyncio
import logging
import websockets
from typing import Optional, Callable, Dict

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

class OKXWebSocketClient:
    """
    Клиент WebSocket для получения данных в реальном времени с биржи OKX.
    Поддерживает подписку на стакан ордеров, тиковые сделки, свечи и др.
    """
    
    # URL WebSocket API для демо и боевого режимов
    WS_URL_PUBLIC = {
        'demo': 'wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999',
        'live': 'wss://ws.okx.com:8443/ws/v5/public'
    }
    
    WS_URL_PRIVATE = {
        'demo': 'wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999',
        'live': 'wss://ws.okx.com:8443/ws/v5/private'
    }
    
    WS_URL_BUSINESS = {
        'demo': 'wss://wspap.okx.com:8443/ws/v5/business?brokerId=9999',
        'live': 'wss://ws.okx.com:8443/ws/v5/business'
    }
    
    def __init__(self, config_path: str = '../config.yaml'):
        """
        Инициализация WebSocket клиента
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        self.mode = self.config.get('mode', 'demo')
        self.symbol = self.config.get('trading', {}).get('symbol', 'ETH-USDT-SWAP')
        
        self.public_ws = None
        self.private_ws = None
        self.business_ws = None
        
        # ID соединений для отслеживания
        self.public_conn_id = None
        self.private_conn_id = None
        self.business_conn_id = None
        
        # Словарь для хранения колбэков при получении данных
        self.callbacks = {
            'orderbook': [],
            'trades': [],
            'candles': [],
            'funding_rate': [],
            'account': [],
            'positions': [],
            'orders': []
        }
        
        # Состояние соединения
        self.is_connected = False
        self.is_authenticated = False
        
        # API ключи
        self.api_key = self.config.get('okx', {}).get(self.mode, {}).get('api_key')
        self.api_secret = self.config.get('okx', {}).get(self.mode, {}).get('api_secret')
        self.password = self.config.get('okx', {}).get(self.mode, {}).get('password')
        
        # Таймеры для ping/pong
        self.ping_interval = 15  # Интервал отправки ping в секундах (менее 30 сек.)
        self.ping_tasks = {}
        
        # Последние метки времени сообщений для каждого соединения
        self.last_message_times: Dict[str, float] = {}
        
        self.logger.info(f"WebSocket клиент OKX инициализирован в режиме {self.mode}")
        
    def _setup_logger(self):
        """Настройка логгера с цветным форматированием"""
        logger = logging.getLogger('okx_websocket')
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
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """
        Генерация подписи для аутентификации в API OKX
        
        Args:
            timestamp: Временная метка в секундах (Unix Epoch time)
            method: HTTP метод (GET/POST)
            request_path: Путь запроса
            body: Тело запроса (для POST запросов)
            
        Returns:
            str: Подпись в формате base64
        """
        if not self.api_secret:
            self.logger.error("Невозможно создать подпись: отсутствует API Secret")
            raise ValueError("API Secret не указан в конфигурации")
            
        # Формируем строку для подписи
        message = timestamp + method + request_path + body
        
        # Создаем HMAC-SHA256 подпись
        mac = hmac.new(
            bytes(self.api_secret, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod='sha256'
        )
        
        # Конвертируем в base64
        signature = base64.b64encode(mac.digest()).decode('utf-8')
        
        return signature
    
    def add_callback(self, channel: str, callback: Callable):
        """
        Добавление колбэка для обработки данных с определенного канала
        
        Args:
            channel: Название канала ('orderbook', 'trades', 'candles', etc.)
            callback: Функция-колбэк, которая будет вызвана при получении данных
        """
        if channel in self.callbacks:
            self.callbacks[channel].append(callback)
            self.logger.info(f"Добавлен обработчик для канала {channel}")
        else:
            self.logger.warning(f"Неизвестный канал: {channel}")
            
    async def connect(self):
        """Установка соединения с WebSocket API"""
        try:
            # Подключаемся к публичному WebSocket API
            self.logger.info(f"Подключение к публичному WebSocket API OKX в режиме {self.mode}")
            self.public_ws = await websockets.connect(self.WS_URL_PUBLIC[self.mode], 
                                                     ping_interval=20, 
                                                     ping_timeout=30)
            
            # Подключаемся к приватному WebSocket API (для аккаунта)
            self.logger.info(f"Подключение к приватному WebSocket API OKX в режиме {self.mode}")
            self.private_ws = await websockets.connect(self.WS_URL_PRIVATE[self.mode], 
                                                      ping_interval=20, 
                                                      ping_timeout=30)
            
            # Подключаемся к business WebSocket API (для свечей)
            self.logger.info(f"Подключение к business WebSocket API OKX в режиме {self.mode}")
            self.business_ws = await websockets.connect(self.WS_URL_BUSINESS[self.mode], 
                                                       ping_interval=20, 
                                                       ping_timeout=30)
            
            # Аутентификация для приватного API
            await self._authenticate()
            
            # Запускаем проверку соединения (ping-pong)
            self.ping_tasks['public'] = asyncio.create_task(self._ping_pong_task('public', self.public_ws))
            self.ping_tasks['private'] = asyncio.create_task(self._ping_pong_task('private', self.private_ws))
            self.ping_tasks['business'] = asyncio.create_task(self._ping_pong_task('business', self.business_ws))
            
            self.is_connected = True
            self.logger.info("Успешное подключение к WebSocket API OKX")
            
        except Exception as e:
            self.logger.error(f"Ошибка подключения к WebSocket: {e}")
            raise
    
    async def _authenticate(self):
        """Аутентификация для приватного WebSocket API"""
        try:
            # Проверяем наличие API ключей
            if not self.api_key or not self.api_secret or not self.password:
                self.logger.error("Невозможно выполнить аутентификацию: отсутствуют API ключи")
                raise ValueError("API ключи не указаны в конфигурации")
                
            # Создаем временную метку в формате Unix (в секундах)
            timestamp = str(int(time.time()))
            
            # Генерируем подпись
            signature = self._generate_signature(timestamp, 'GET', '/users/self/verify', '')
            
            # Создаем сообщение для аутентификации
            auth_message = {
                "op": "login",
                "args": [
                    {
                        "apiKey": self.api_key,
                        "passphrase": self.password,
                        "timestamp": timestamp,
                        "sign": signature
                    }
                ]
            }
            
            # Отправляем сообщение с аутентификацией
            await self.private_ws.send(json.dumps(auth_message))
            
            # Получаем ответ
            response = await self.private_ws.recv()
            response_data = json.loads(response)
            
            # Обновляем время последнего сообщения для приватного соединения
            self.last_message_times['private'] = time.time()
            
            # Проверяем результат аутентификации
            if response_data.get('event') == 'login' and response_data.get('code') == '0':
                self.is_authenticated = True
                self.private_conn_id = response_data.get('connId', '')
                self.logger.info(f"Успешная аутентификация на приватном WebSocket API с connId: {self.private_conn_id}")
            else:
                self.logger.error(f"Ошибка аутентификации: {response_data}")
                raise Exception(f"Ошибка аутентификации: {response_data.get('msg', 'Неизвестная ошибка')}")
                
        except Exception as e:
            self.logger.error(f"Ошибка при аутентификации: {e}")
            raise
    
    async def _ping_pong_task(self, ws_name: str, ws):
        """
        Задача для поддержания соединения активным с помощью ping/pong
        
        Args:
            ws_name: Название WebSocket соединения для логирования
            ws: WebSocket соединение
        """
        try:
            while True:
                # Проверяем время с последнего полученного сообщения
                current_time = time.time()
                last_message_time = self.last_message_times.get(ws_name, current_time)
                
                # Если прошло больше ping_interval секунд с последнего сообщения,
                # отправляем ping для поддержания соединения
                if current_time - last_message_time > self.ping_interval:
                    self.logger.debug(f"Отправка ping для {ws_name} WebSocket")
                    await ws.send('ping')
                    
                    # Ожидаем ответа pong с таймаутом
                    try:
                        pong_response = await asyncio.wait_for(ws.recv(), timeout=5)
                        if pong_response == 'pong':
                            self.logger.debug(f"Получен pong ответ от {ws_name} WebSocket")
                            self.last_message_times[ws_name] = time.time()
                        else:
                            # Если получили не pong, обрабатываем как обычное сообщение
                            message_data = json.loads(pong_response)
                            await self._handle_message(message_data)
                            self.last_message_times[ws_name] = time.time()
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Не получен pong ответ от {ws_name} WebSocket, возможно соединение потеряно")
                        # Можно добавить логику переподключения здесь
                
                # Ждем некоторое время перед следующей проверкой
                await asyncio.sleep(5)
                
        except asyncio.CancelledError:
            self.logger.info(f"Ping/pong задача для {ws_name} WebSocket отменена")
        except Exception as e:
            self.logger.error(f"Ошибка в ping/pong задаче для {ws_name} WebSocket: {e}")
    
    async def subscribe_orderbook(self, symbol: Optional[str] = None, depth: str = "books", depth_size: int = 400):
        """
        Подписка на обновления стакана ордеров
        
        Args:
            symbol: Символ торговой пары (если None, используется self.symbol)
            depth: Тип стакана ("books" для 400 уровней, "books5" для 5 уровней, "books50-l2-tbt" для 50 уровней tick-by-tick)
            depth_size: Глубина стакана (количество уровней цен)
        """
        symbol = symbol or self.symbol
        
        # Проверяем, не передали ли числовое значение вместо названия канала
        if isinstance(depth, (int, str)) and str(depth).isdigit():
            depth_size = int(depth)
            depth = "books"
        
        # Проверка на допустимые значения типа стакана
        valid_channels = ["books", "books5", "books50-l2-tbt", "books-l2-tbt"]
        if depth not in valid_channels:
            self.logger.warning(f"Неподдерживаемый тип стакана: {depth}. Используем стандартный 'books'")
            channel = "books"
        else:
            channel = depth
        
        # Добавляем параметры instType и sz как в okx_orderbook_debug
        args = {
            "instId": symbol,
            "instType": "SWAP",
            "sz": depth_size  # Явное указание глубины стакана
        }
        
        await self._subscribe(self.public_ws, channel, args)
        self.logger.info(f"Подписка на стакан ордеров {channel} для {symbol} с глубиной {depth_size}")
    
    async def subscribe_trades(self, symbol: Optional[str] = None):
        """
        Подписка на тиковые сделки
        
        Args:
            symbol: Символ торговой пары (если None, используется self.symbol)
        """
        symbol = symbol or self.symbol
        channel = "trades"
        
        await self._subscribe(self.public_ws, channel, {"instId": symbol})
        self.logger.info(f"Подписка на тиковые сделки для {symbol}")
    
    async def subscribe_candles(self, symbol: Optional[str] = None, timeframe: str = "1m"):
        """
        Подписка на свечные данные
        
        Args:
            symbol: Символ торговой пары (если None, используется self.symbol)
            timeframe: Таймфрейм свечей ("1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "12H", "1D", "2D", "3D", "1W", "1M", "3M")
        """
        symbol = symbol or self.symbol
        
        # Проверка допустимых таймфреймов и преобразование в формат OKX
        valid_timeframes = {
            "1m": "1m", 
            "3m": "3m", 
            "5m": "5m", 
            "15m": "15m", 
            "30m": "30m", 
            "1H": "1H", 
            "2H": "2H", 
            "4H": "4H", 
            "6H": "6H", 
            "12H": "12H", 
            "1D": "1D", 
            "2D": "2D", 
            "3D": "3D", 
            "1W": "1W", 
            "1M": "1M", 
            "3M": "3M"
        }
        
        if timeframe not in valid_timeframes:
            self.logger.warning(f"Неподдерживаемый таймфрейм: {timeframe}. Используем стандартный '1m'")
            timeframe = "1m"
        
        # Для канала свечей используем формат API OKX
        okx_timeframe = valid_timeframes[timeframe]
        channel = f"candle{okx_timeframe}"
        
        # Используем business endpoint для свечей как в okx_orderbook_debug и добавляем instType
        args = {
            "instId": symbol,
            "instType": "SWAP"
        }
        
        # Используем business_ws вместо public_ws для свечей
        if not self.business_ws:
            self.logger.info(f"Подключение к business WebSocket API OKX в режиме {self.mode}")
            self.business_ws = await websockets.connect(self.WS_URL_BUSINESS[self.mode], 
                                                      ping_interval=20, 
                                                      ping_timeout=30)
            # Запускаем проверку соединения (ping-pong)
            self.ping_tasks['business'] = asyncio.create_task(self._ping_pong_task('business', self.business_ws))
        
        await self._subscribe(self.business_ws, channel, args)
        self.logger.info(f"Подписка на свечи {timeframe} для {symbol}")
    
    async def subscribe_funding_rate(self, symbol: Optional[str] = None):
        """
        Подписка на обновления funding rate
        
        Args:
            symbol: Символ торговой пары (если None, используется self.symbol)
        """
        symbol = symbol or self.symbol
        channel = "funding-rate"
        
        await self._subscribe(self.public_ws, channel, {"instId": symbol})
        self.logger.info(f"Подписка на обновления funding rate для {symbol}")
    
    async def subscribe_account(self):
        """Подписка на обновления аккаунта (требует аутентификации)"""
        if not self.is_authenticated:
            self.logger.error("Невозможно подписаться на обновления аккаунта: отсутствует аутентификация")
            return
            
        channel = "account"
        
        await self._subscribe(self.private_ws, channel, {})
        self.logger.info(f"Подписка на обновления аккаунта")

    async def subscribe_positions(self, inst_type: str = "ANY"):
        """
        Подписка на обновления позиций (требует аутентификации)
        
        Args:
            inst_type: Тип инструмента ("SPOT", "MARGIN", "SWAP", "FUTURES", "OPTION", "ANY")
        """
        if not self.is_authenticated:
            self.logger.error("Невозможно подписаться на обновления позиций: отсутствует аутентификация")
            return
            
        channel = "positions"
        
        await self._subscribe(self.private_ws, channel, {"instType": inst_type})
        self.logger.info(f"Подписка на обновления позиций для типа инструмента {inst_type}")
        
    async def subscribe_orders(self, inst_type: str = "ANY", inst_family: Optional[str] = None, inst_id: Optional[str] = None):
        """
        Подписка на обновления ордеров (требует аутентификации)
        
        Args:
            inst_type: Тип инструмента ("SPOT", "MARGIN", "SWAP", "FUTURES", "OPTION", "ANY")
            inst_family: Семейство инструментов (опционально)
            inst_id: ID инструмента (опционально)
        """
        if not self.is_authenticated:
            self.logger.error("Невозможно подписаться на обновления ордеров: отсутствует аутентификация")
            return
            
        channel = "orders"
        args = {"instType": inst_type}
        
        if inst_family:
            args["instFamily"] = inst_family
        if inst_id:
            args["instId"] = inst_id
            
        await self._subscribe(self.private_ws, channel, args)
        details = f"для типа инструмента {inst_type}"
        if inst_family:
            details += f", семейства {inst_family}"
        if inst_id:
            details += f", инструмента {inst_id}"
            
        self.logger.info(f"Подписка на обновления ордеров {details}")
    
    async def _subscribe(self, ws, channel: str, args: dict):
        """
        Отправка запроса на подписку на канал
        
        Args:
            ws: WebSocket соединение
            channel: Название канала
            args: Дополнительные аргументы для подписки
        """
        # Создаем аргументы подписки
        subscription_args = {"channel": channel}
        subscription_args.update(args)
        
        # Логируем запрос подписки для отладки
        self.logger.debug(f"Отправка запроса на подписку: {subscription_args}")
        
        subscribe_message = {
            "op": "subscribe",
            "args": [subscription_args]
        }
        
        await ws.send(json.dumps(subscribe_message))
    
    async def _unsubscribe(self, ws, channel: str, args: dict):
        """
        Отправка запроса на отписку от канала
        
        Args:
            ws: WebSocket соединение
            channel: Название канала
            args: Дополнительные аргументы для отписки
        """
        # Создаем аргументы отписки
        unsubscription_args = {"channel": channel}
        unsubscription_args.update(args)
        
        unsubscribe_message = {
            "op": "unsubscribe",
            "args": [unsubscription_args]
        }
        
        await ws.send(json.dumps(unsubscribe_message))
    
    async def _handle_message(self, message_data: dict):
        """
        Обработка входящего сообщения и вызов соответствующих колбэков
        
        Args:
            message_data: Данные сообщения
        """
        try:
            # Определяем тип сообщения
            if 'event' in message_data:
                # Это событие (подписка, отписка, ошибка)
                event = message_data.get('event')
                
                # Сохраняем ID соединения, если это сообщение о подписке
                if 'connId' in message_data:
                    conn_id = message_data.get('connId')
                    if event == 'subscribe' and 'arg' in message_data:
                        channel = message_data['arg'].get('channel')
                        self.logger.info(f"Успешная подписка на канал {channel} с connId: {conn_id}")
                
                if event == 'error':
                    self.logger.error(f"Ошибка WebSocket: {message_data}")
                elif event == 'subscribe':
                    self.logger.info(f"Подписка на канал: {message_data.get('arg', {}).get('channel')}")
                elif event == 'unsubscribe':
                    self.logger.info(f"Отписка от канала: {message_data.get('arg', {}).get('channel')}")
                elif event == 'login':
                    self.logger.info(f"Событие логина: {message_data}")
                elif event == 'channel-conn-count':
                    channel = message_data.get('channel')
                    conn_count = message_data.get('connCount')
                    conn_id = message_data.get('connId')
                    self.logger.info(f"Количество соединений для канала {channel}: {conn_count}, connId: {conn_id}")
                elif event == 'channel-conn-count-error':
                    channel = message_data.get('channel')
                    conn_count = message_data.get('connCount')
                    conn_id = message_data.get('connId')
                    self.logger.error(f"Ошибка превышения лимита соединений для канала {channel}: {conn_count}, connId: {conn_id}")
                elif event == 'notice':
                    code = message_data.get('code')
                    msg = message_data.get('msg')
                    conn_id = message_data.get('connId')
                    self.logger.warning(f"Уведомление WebSocket: {msg}, код: {code}, connId: {conn_id}")
                else:
                    self.logger.info(f"Событие WebSocket: {event}")
            
            elif 'data' in message_data:
                # Это данные канала
                channel = message_data.get('arg', {}).get('channel', '')
                
                # Определяем, к какому типу данных относится сообщение
                channel_type = None
                if channel.startswith('books'):
                    channel_type = 'orderbook'
                elif channel == 'trades':
                    channel_type = 'trades'
                elif channel.startswith('candle'):
                    channel_type = 'candles'
                elif channel == 'funding-rate':
                    channel_type = 'funding_rate'
                elif channel == 'account':
                    channel_type = 'account'
                elif channel == 'positions':
                    channel_type = 'positions'
                elif channel == 'orders':
                    channel_type = 'orders'
                
                # Если есть колбэки для данного типа канала, вызываем их
                if channel_type and channel_type in self.callbacks:
                    for callback in self.callbacks[channel_type]:
                        asyncio.create_task(callback(message_data['data']))
        
        except Exception as e:
            self.logger.error(f"Ошибка обработки сообщения: {e}")
    
    async def listen(self):
        """Прослушивание WebSocket соединений и обработка сообщений"""
        if not self.is_connected:
            self.logger.error("Невозможно начать прослушивание: отсутствует соединение")
            return
            
        try:
            # Запускаем задачи для прослушивания WebSocket соединений
            public_task = asyncio.create_task(self._listen_ws('public', self.public_ws))
            private_task = asyncio.create_task(self._listen_ws('private', self.private_ws))
            business_task = asyncio.create_task(self._listen_ws('business', self.business_ws))
            
            # Ждем завершения всех задач
            await asyncio.gather(public_task, private_task, business_task)
        except KeyboardInterrupt:
            self.logger.info("Прослушивание WebSocket остановлено пользователем")
        except asyncio.CancelledError:
            self.logger.info("Прослушивание WebSocket отменено")
        except Exception as e:
            self.logger.error(f"Ошибка при прослушивании WebSocket: {e}")
        finally:
            # При любой ошибке закрываем соединения
            await self.close()
    
    async def _listen_ws(self, ws_name: str, ws):
        """
        Прослушивание отдельного WebSocket соединения
        
        Args:
            ws_name: Название WebSocket соединения для логирования
            ws: WebSocket соединение
        """
        try:
            # Инициализируем время последнего сообщения
            self.last_message_times[ws_name] = time.time()
            
            while True:
                message = await ws.recv()
                
                # Обновляем время последнего полученного сообщения
                self.last_message_times[ws_name] = time.time()
                
                # Обрабатываем ping/pong сообщения
                if message == 'ping':
                    await ws.send('pong')
                    self.logger.debug(f"Получен ping от {ws_name}, отправлен pong")
                    continue
                elif message == 'pong':
                    self.logger.debug(f"Получен pong от {ws_name}")
                    continue
                
                # Обрабатываем обычные JSON сообщения
                try:
                    message_data = json.loads(message)
                    await self._handle_message(message_data)
                except json.JSONDecodeError:
                    self.logger.warning(f"Получено не-JSON сообщение от {ws_name}: {message}")
                
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.warning(f"WebSocket соединение {ws_name} закрыто: {e}")
            self.is_connected = False
            # Можно добавить логику переподключения здесь
        except KeyboardInterrupt:
            self.logger.info(f"Прослушивание WebSocket соединения {ws_name} остановлено пользователем")
            return
        except asyncio.CancelledError:
            self.logger.info(f"Прослушивание WebSocket соединения {ws_name} отменено")
            return
        except Exception as e:
            self.logger.error(f"Ошибка в процессе прослушивания WebSocket {ws_name}: {e}")
            self.is_connected = False
    
    async def close(self):
        """Закрытие WebSocket соединений"""
        try:
            # Отменяем задачи ping/pong
            for task_name, task in self.ping_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            self.ping_tasks.clear()
            
            # Закрываем соединения
            if self.public_ws:
                await self.public_ws.close()
                
            if self.private_ws:
                await self.private_ws.close()
                
            if self.business_ws:
                await self.business_ws.close()
                
            self.is_connected = False
            self.is_authenticated = False
            self.logger.info("WebSocket соединения закрыты")
            
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии WebSocket соединений: {e}")

# Функция для тестирования WebSocket клиента
async def test_websocket():
    """Тестирование функциональности WebSocket клиента"""
    # Создаем экземпляр WebSocket клиента
    ws_client = OKXWebSocketClient(config_path='../config.yaml')
    
    # Определяем обработчик для стакана ордеров
    async def orderbook_handler(data):
        print(f"Получено обновление стакана: {len(data[0]['asks'])} заявок на продажу, {len(data[0]['bids'])} заявок на покупку")
        
    # Определяем обработчик для тиковых сделок
    async def trades_handler(data):
        print(f"Получены сделки: {len(data)} сделок")
        for trade in data:
            print(f"  {trade['side']} {trade['sz']} по цене {trade['px']}")
    
    # Добавляем обработчики
    ws_client.add_callback('orderbook', orderbook_handler)
    ws_client.add_callback('trades', trades_handler)
    
    # Устанавливаем соединение
    await ws_client.connect()
    
    # Подписываемся на каналы
    await ws_client.subscribe_orderbook()
    await ws_client.subscribe_trades()
    
    # Слушаем некоторое время
    try:
        print("Прослушивание WebSocket сообщений в течение 30 секунд...")
        listen_task = asyncio.create_task(ws_client.listen())
        await asyncio.sleep(30)
    finally:
        # Закрываем соединение
        await ws_client.close()

if __name__ == "__main__":
    # Запускаем тест
    asyncio.run(test_websocket())