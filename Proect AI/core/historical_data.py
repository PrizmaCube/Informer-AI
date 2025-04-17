import os
import sys
import asyncio
import pandas as pd
import sqlite3
import ccxt.async_support as ccxt
import logging
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Добавляем родительскую директорию в путь для импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Настройка логирования
from core.exchange import ColoredFormatter

class HistoricalDataLoader:
    """
    Класс для загрузки исторических данных с биржи OKX
    и сохранения их в SQLite базу данных
    """
    
    def __init__(self, config_path: str = 'config.yaml', db_path: str = 'data/db.sqlite'):
        """
        Инициализирует загрузчик исторических данных
        
        Args:
            config_path: Путь к файлу конфигурации
            db_path: Путь к файлу базы данных SQLite
        """
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        self.mode = self.config.get('mode', 'demo')
        self.symbol = self.config.get('trading', {}).get('symbol', 'ETH-USDT-SWAP')
        
        # Путь к базе данных
        self.db_path = db_path
        
        # Обеспечиваем наличие директории для БД
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Инициализируем подключение к бирже
        self.exchange = None
        self._initialize_exchange()
        
        self.logger.info(f"HistoricalDataLoader инициализирован в режиме {self.mode}")
        
    def _setup_logger(self):
        """Настройка логгера с цветным форматированием"""
        logger = logging.getLogger('historical_data')
        logger.setLevel(logging.INFO)
        
        # Проверяем, есть ли уже обработчики у логгера
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            colored_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(colored_formatter)
            
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
            if not self.config.get('okx', {}).get(self.mode, {}).get('api_key') or \
               not self.config.get('okx', {}).get(self.mode, {}).get('api_secret') or \
               not self.config.get('okx', {}).get(self.mode, {}).get('password'):
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
                self.exchange.urls = {
                    'api': {
                        'rest': 'https://www.okx.com',
                        'ws': 'wss://wspap.okx.com:8443/ws/v5'
                    }
                }
                
            # Устанавливаем параметры запросов    
            self.exchange.options['recvWindow'] = 60000  # Длительный период для исторических данных
            
            self.logger.info(f"Подключение к бирже OKX ({self.mode}) инициализировано")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации биржи: {e}")
            raise
    
    def _create_tables(self):
        """Создает необходимые таблицы в базе данных, если они не существуют"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица для хранения OHLCV данных
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                timeframe TEXT NOT NULL,
                UNIQUE(symbol, timestamp, timeframe)
            )
            ''')
            
            # Индекс для ускорения поиска
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe_timestamp 
            ON candles(symbol, timeframe, timestamp)
            ''')
            
            # Таблица для хранения тиковых сделок
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                side TEXT NOT NULL,
                trade_id TEXT,
                UNIQUE(symbol, trade_id)
            )
            ''')
            
            # Индекс для ускорения поиска
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp 
            ON trades(symbol, timestamp)
            ''')
            
            # Таблица для хранения funding rate
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS funding_rate (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                rate REAL NOT NULL,
                next_funding_time INTEGER,
                UNIQUE(symbol, timestamp)
            )
            ''')
            
            # Индекс для ускорения поиска
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_funding_rate_symbol_timestamp 
            ON funding_rate(symbol, timestamp)
            ''')
            
            conn.commit()
            self.logger.info("Таблицы в базе данных созданы успешно")
            
        except Exception as e:
            self.logger.error(f"Ошибка при создании таблиц: {e}")
            conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    async def fetch_ohlcv_history(self, symbol: str, timeframe: str, 
                                 since: Optional[int] = None, 
                                 limit: int = 1000,
                                 days_back: int = 30) -> List[List]:
        """
        Загружает исторические OHLCV данные для указанного символа и таймфрейма
        
        Args:
            symbol: Символ торговой пары
            timeframe: Таймфрейм свечей (1m, 5m, 15m, 1h, 4h, 1d)
            since: Начальная временная метка в миллисекундах (если None, используется days_back)
            limit: Максимальное количество свечей в одном запросе
            days_back: Количество дней для загрузки данных (если since не указан)
            
        Returns:
            List[List]: Список свечей в формате [timestamp, open, high, low, close, volume]
        """
        try:
            if not since:
                # Если начальная дата не указана, используем days_back
                since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            all_candles = []
            current_since = since
            
            while True:
                self.logger.info(f"Загрузка OHLCV для {symbol} ({timeframe}), начиная с {datetime.fromtimestamp(current_since/1000)}")
                
                # Запрашиваем свечи
                candles = await self.exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                
                # Для следующего запроса устанавливаем временную метку последней свечи + 1 минута
                if len(candles) < limit:
                    # Достигнут конец данных
                    break
                    
                # Используем временную метку последней свечи + 1 мс для следующего запроса
                current_since = candles[-1][0] + 1
                
                # Проверяем, не достигли ли мы текущего времени
                if current_since >= int(datetime.now().timestamp() * 1000):
                    break
                    
                # Добавляем задержку, чтобы не превысить лимиты API
                await asyncio.sleep(1)
            
            self.logger.info(f"Загружено {len(all_candles)} свечей {timeframe} для {symbol}")
            return all_candles
        
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке OHLCV данных: {e}")
            raise
    
    async def save_ohlcv_to_db(self, symbol: str, timeframe: str, candles: List[List]) -> int:
        """
        Сохраняет OHLCV данные в базу данных
        
        Args:
            symbol: Символ торговой пары
            timeframe: Таймфрейм свечей
            candles: Список свечей в формате [timestamp, open, high, low, close, volume]
            
        Returns:
            int: Количество добавленных записей
        """
        if not candles:
            return 0
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Убеждаемся, что таблицы существуют
            self._create_tables()
            
            # Подготавливаем данные для вставки
            rows_to_insert = []
            for candle in candles:
                timestamp, open_price, high, low, close, volume = candle
                rows_to_insert.append((
                    symbol, 
                    timestamp, 
                    open_price, 
                    high, 
                    low, 
                    close, 
                    volume,
                    timeframe
                ))
            
            # Вставляем данные с обработкой дубликатов
            cursor.executemany('''
            INSERT OR IGNORE INTO candles 
            (symbol, timestamp, open, high, low, close, volume, timeframe)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', rows_to_insert)
            
            inserted_count = cursor.rowcount
            conn.commit()
            
            self.logger.info(f"Сохранено {inserted_count} новых свечей {timeframe} для {symbol} в базу данных")
            return inserted_count
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении OHLCV данных в базу: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    async def fetch_funding_rate_history(self, symbol: str, days_back: int = 30) -> List[Dict]:
        """
        Загружает историю ставок финансирования для указанного символа
        
        Args:
            symbol: Символ торговой пары
            days_back: Количество дней для загрузки данных
            
        Returns:
            List[Dict]: Список ставок финансирования
        """
        try:
            # Для получения истории funding rate нужно использовать OKX API напрямую
            # CCXT не имеет стандартного метода для этого
            
            # Преобразуем символ для OKX API
            instrument_id = symbol
            
            # Примерный код запроса к OKX API (требует доработки)
            since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            # Здесь должен быть запрос к OKX API для получения истории funding rate
            # Но поскольку CCXT не имеет прямого метода, придется использовать
            # нативный API OKX
            
            # Пример для OKX через кастомный запрос CCXT
            funding_history = []
            
            # Симулируем результат
            self.logger.warning("Функция получения истории funding rate требует доработки с использованием нативного API OKX")
            self.logger.info(f"Запрашиваем историю funding rate для {symbol} за последние {days_back} дней")
            
            # Возвращаем пустой список, т.к. нужно доработать метод
            return funding_history
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке истории funding rate: {e}")
            raise
    
    async def save_funding_rate_to_db(self, symbol: str, funding_rates: List[Dict]) -> int:
        """
        Сохраняет данные о ставках финансирования в базу данных
        
        Args:
            symbol: Символ торговой пары
            funding_rates: Список ставок финансирования
            
        Returns:
            int: Количество добавленных записей
        """
        if not funding_rates:
            return 0
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Убеждаемся, что таблицы существуют
            self._create_tables()
            
            # Подготавливаем данные для вставки
            rows_to_insert = []
            for rate in funding_rates:
                timestamp = rate.get('timestamp')
                funding_rate = rate.get('rate')
                next_funding_time = rate.get('next_funding_time', None)
                
                rows_to_insert.append((
                    symbol,
                    timestamp,
                    funding_rate,
                    next_funding_time
                ))
            
            # Вставляем данные с обработкой дубликатов
            cursor.executemany('''
            INSERT OR IGNORE INTO funding_rate 
            (symbol, timestamp, rate, next_funding_time)
            VALUES (?, ?, ?, ?)
            ''', rows_to_insert)
            
            inserted_count = cursor.rowcount
            conn.commit()
            
            self.logger.info(f"Сохранено {inserted_count} записей о ставках финансирования для {symbol} в базу данных")
            return inserted_count
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении данных о ставках финансирования в базу: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def get_ohlcv_from_db(self, symbol: str, timeframe: str, 
                          start_time: Optional[int] = None, 
                          end_time: Optional[int] = None,
                          limit: Optional[int] = None) -> pd.DataFrame:
        """
        Получает OHLCV данные из базы данных
        
        Args:
            symbol: Символ торговой пары
            timeframe: Таймфрейм свечей
            start_time: Начальная временная метка в миллисекундах
            end_time: Конечная временная метка в миллисекундах
            limit: Ограничение количества возвращаемых свечей
            
        Returns:
            pd.DataFrame: DataFrame с OHLCV данными
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT timestamp, open, high, low, close, volume FROM candles WHERE symbol = ? AND timeframe = ?"
            params = [symbol, timeframe]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
                
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
                
            query += " ORDER BY timestamp"
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Загружаем данные в pandas DataFrame
            df = pd.read_sql_query(query, conn, params=params)
            
            # Преобразуем timestamp в datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Устанавливаем datetime в качестве индекса
            df.set_index('datetime', inplace=True)
            
            self.logger.info(f"Загружено {len(df)} свечей {timeframe} для {symbol} из базы данных")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении OHLCV данных из базы: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    async def load_and_save_history(self, symbol: str, timeframes: List[str], days_back: int = 30) -> Dict[str, int]:
        """
        Загружает и сохраняет исторические данные для заданного символа и таймфреймов
        
        Args:
            symbol: Символ торговой пары
            timeframes: Список таймфреймов для загрузки
            days_back: Количество дней для загрузки данных
            
        Returns:
            Dict[str, int]: Словарь с количеством добавленных записей по каждому таймфрейму
        """
        results = {}
        
        for timeframe in timeframes:
            try:
                # Загружаем данные
                candles = await self.fetch_ohlcv_history(symbol, timeframe, days_back=days_back)
                
                # Сохраняем в базу
                inserted = await self.save_ohlcv_to_db(symbol, timeframe, candles)
                
                results[timeframe] = inserted
                
            except Exception as e:
                self.logger.error(f"Ошибка при загрузке и сохранении данных для {symbol} ({timeframe}): {e}")
                results[timeframe] = -1
        
        # Загружаем и сохраняем funding rate
        try:
            funding_rates = await self.fetch_funding_rate_history(symbol, days_back=days_back)
            
            inserted = await self.save_funding_rate_to_db(symbol, funding_rates)
            
            results['funding_rate'] = inserted
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке и сохранении funding rate для {symbol}: {e}")
            results['funding_rate'] = -1
        
        return results
    
    async def close(self):
        """Закрывает подключение к бирже"""
        if self.exchange:
            await self.exchange.close()
            self.logger.info("Подключение к бирже закрыто")

# Функция для тестирования модуля
async def test_historical_data():
    """
    Тестирование загрузки исторических данных
    """
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ ЗАГРУЗКИ ИСТОРИЧЕСКИХ ДАННЫХ")
    print("=" * 50)
    
    # Создаем экземпляр загрузчика
    data_loader = HistoricalDataLoader(config_path='config.yaml', db_path='data/test_db.sqlite')
    
    try:
        # Загружаем данные для одного символа и нескольких таймфреймов
        symbol = data_loader.symbol
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        days_back = 7  # Тестовый период - 7 дней
        
        print(f"Загрузка данных для {symbol}, таймфреймы: {', '.join(timeframes)}, период: {days_back} дней")
        
        # Загружаем и сохраняем данные
        results = await data_loader.load_and_save_history(symbol, timeframes, days_back)
        
        print("\nРезультаты:")
        for tf, count in results.items():
            print(f"  {tf}: {'Ошибка' if count == -1 else f'добавлено {count} записей'}")
        
        # Тестируем получение данных из базы
        for tf in timeframes:
            df = data_loader.get_ohlcv_from_db(symbol, tf, limit=5)
            print(f"\nПример данных для {tf}:")
            print(df.head())
            
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
    finally:
        # Закрываем подключение
        await data_loader.close()
        print("\nТест завершен")
        print("=" * 50)

if __name__ == "__main__":
    # Запускаем тест
    try:
        asyncio.run(test_historical_data())
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}") 