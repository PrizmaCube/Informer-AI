import os
import yaml
import asyncio
import logging

# Импортируем основные компоненты
from core.exchange import OKXExchange
from core.websocket_client import OKXWebSocketClient

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

class InformerTrading:
    """
    Основной класс приложения для торговли ETH/USDT фьючерсами на OKX
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Инициализация основного класса приложения
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        # Настройка логгера
        self.logger = self._setup_logger()
        self.logger.info("Инициализация приложения InformerTrading")
        
        # Загрузка конфигурации
        self.config = self._load_config(config_path)
        self.mode = self.config.get('mode', 'demo')
        self.logger.info(f"Режим приложения: {self.mode}")
        
        # Инициализация компонентов
        self.exchange = None
        self.ws_client = None
        
        # Обработчики данных (будут добавлены позже)
        self.data_handlers = {}
        
        self.logger.info("Приложение InformerTrading инициализировано")
    
    def _setup_logger(self):
        """Настройка логгера с цветным форматированием"""
        # Создаем директорию для логов, если она не существует
        os.makedirs('data/logs', exist_ok=True)
        
        logger = logging.getLogger('informer_trading')
        logger.setLevel(logging.INFO)
        
        # Проверяем, есть ли уже обработчики у логгера
        if not logger.handlers:
            # Обработчик для вывода в консоль с цветным форматированием
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Обработчик для записи в файл
            file_handler = logging.FileHandler('data/logs/trading_terminal.log', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # Формат логов с цветами для консоли
            colored_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(colored_formatter)
            
            # Обычный формат для файла
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            
            # Добавляем обработчики к логгеру
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
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
    
    async def initialize_components(self):
        """Инициализация компонентов системы"""
        try:
            self.logger.info("Инициализация биржи и WebSocket клиента")
            
            # Инициализация REST API клиента
            self.exchange = OKXExchange(config_path='config.yaml')
            
            # Инициализация WebSocket клиента
            self.ws_client = OKXWebSocketClient(config_path='config.yaml')
            
            self.logger.info("Компоненты успешно инициализированы")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации компонентов: {e}")
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
        
        # Обновляем режим в конфигурации
        self.mode = new_mode
        self.config['mode'] = new_mode
        
        # Сохраняем обновленную конфигурацию
        self._save_config()
        
        # Переключаем режим в компонентах
        if self.exchange:
            self.exchange.switch_mode(new_mode)
        
        self.logger.info(f"Успешно переключено на режим {new_mode}")
    
    def _save_config(self):
        """Сохранение конфигурации в файл"""
        try:
            with open('config.yaml', 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            self.logger.info("Конфигурация успешно сохранена")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения конфигурации: {e}")
    
    async def connect(self):
        """Установка соединения с биржей"""
        try:
            # Подключение WebSocket
            if self.ws_client:
                await self.ws_client.connect()
            
            self.logger.info("Соединение с биржей установлено")
            
        except Exception as e:
            self.logger.error(f"Ошибка подключения к бирже: {e}")
            raise
    
    async def subscribe_to_market_data(self):
        """Подписка на рыночные данные"""
        try:
            if not self.ws_client:
                self.logger.error("WebSocket клиент не инициализирован")
                return
                
            # Подписываемся на стакан ордеров с глубиной 400 уровней
            await self.ws_client.subscribe_orderbook(depth="400")
            
            # Подписываемся на тиковые сделки
            await self.ws_client.subscribe_trades()
            
            # Подписываемся на свечные данные для разных таймфреймов
            await self.ws_client.subscribe_candles(timeframe="1m")
            await self.ws_client.subscribe_candles(timeframe="5m")
            
            # Подписываемся на обновления funding rate
            await self.ws_client.subscribe_funding_rate()
            
            # Подписываемся на обновления аккаунта
            await self.ws_client.subscribe_account()
            
            self.logger.info("Подписка на все каналы рыночных данных выполнена")
            
        except Exception as e:
            self.logger.error(f"Ошибка подписки на рыночные данные: {e}")
            raise
    
    async def run(self):
        """Запуск основного цикла приложения"""
        try:
            # Инициализация компонентов
            await self.initialize_components()
            
            # Устанавливаем соединение
            await self.connect()
            
            # Подписываемся на рыночные данные
            await self.subscribe_to_market_data()
            
            # Запускаем прослушивание WebSocket
            if self.ws_client:
                await self.ws_client.listen()
            
        except KeyboardInterrupt:
            self.logger.info("Приложение остановлено пользователем")
        except Exception as e:
            self.logger.error(f"Ошибка в основном цикле: {e}")
        finally:
            # Закрываем соединения
            if self.ws_client:
                await self.ws_client.close()
            
            self.logger.info("Завершение работы приложения")

    async def shutdown(self):
        """Корректное завершение работы приложения"""
        try:
            # Закрываем WebSocket соединения
            if self.ws_client:
                await self.ws_client.close()
                
            self.logger.info("Ресурсы освобождены")
        except Exception as e:
            self.logger.error(f"Ошибка при завершении работы: {e}")

async def main():
    """Функция запуска приложения"""
    app = InformerTrading(config_path='config.yaml')
    
    try:
        # Запускаем приложение
        await app.run()
    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания (Ctrl+C)")
    except Exception as e:
        print(f"Критическая ошибка: {e}")
    finally:
        # Корректное закрытие ресурсов
        await app.shutdown()

if __name__ == "__main__":
    try:
        # Используем asyncio.run с обработкой KeyboardInterrupt
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем (Ctrl+C)")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}") 