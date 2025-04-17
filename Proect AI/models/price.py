import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import logging
import yaml
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Добавляем родительскую директорию в путь для импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.exchange import ColoredFormatter

# Проверка доступности CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PriceDataset(Dataset):
    """
    Датасет для обучения модели прогнозирования цены
    """
    def __init__(self, X, y):
        """
        Инициализация датасета
        
        Args:
            X: Входные признаки (numpy array)
            y: Целевые значения (numpy array)
        """
        self.X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTM_PricePredictor(nn.Module):
    """
    LSTM модель для прогнозирования цены
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        """
        Инициализация LSTM модели
        
        Args:
            input_size: Количество входных признаков
            hidden_size: Размер скрытого слоя
            num_layers: Количество слоев LSTM
            output_size: Количество выходных значений (обычно 1 для прогноза цены)
            dropout: Вероятность dropout для регуляризации
        """
        super(LSTM_PricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM слой
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(dropout)
        
        # Полносвязный слой для преобразования выхода LSTM в прогноз
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Прямое распространение
        
        Args:
            x: Входной тензор [batch_size, sequence_length, input_size]
            
        Returns:
            torch.Tensor: Прогноз [batch_size, output_size]
        """
        # Инициализация скрытого состояния и ячейки
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Прямое распространение через LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Берем выход последнего временного шага
        out = out[:, -1, :]
        
        # Применяем dropout для регуляризации
        out = self.dropout(out)
        
        # Преобразуем в выходное значение
        out = self.fc(out)
        
        return out

class PricePredictor:
    """
    Класс для обучения и использования модели прогнозирования цены
    """
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Инициализация прогнозатора цены
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        
        # Параметры модели
        self.input_size = self.config.get('model', {}).get('price', {}).get('input_size', 15)
        self.hidden_size = self.config.get('model', {}).get('price', {}).get('hidden_size', 128)
        self.num_layers = self.config.get('model', {}).get('price', {}).get('num_layers', 2)
        self.dropout = self.config.get('model', {}).get('price', {}).get('dropout', 0.2)
        self.seq_length = self.config.get('model', {}).get('price', {}).get('seq_length', 30)
        self.output_size = self.config.get('model', {}).get('price', {}).get('output_size', 1)
        self.prediction_horizon = self.config.get('model', {}).get('price', {}).get('prediction_horizon', 1)
        
        # Параметры обучения
        self.batch_size = self.config.get('training', {}).get('batch_size', 64)
        self.learning_rate = self.config.get('training', {}).get('learning_rate', 0.001)
        self.num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        
        # Пути к файлам весов и скейлеров
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'weights')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Создаем модель
        self.model = LSTM_PricePredictor(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            dropout=self.dropout
        ).to(DEVICE)
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Скейлеры для нормализации данных
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        self.logger.info("Модель прогнозирования цены инициализирована")
        self.logger.info(f"Устройство: {DEVICE}")
        
    def _setup_logger(self):
        """Настройка логгера с цветным форматированием"""
        logger = logging.getLogger('price_predictor')
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
    
    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str], target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения/прогнозирования
        
        Args:
            df: DataFrame с данными OHLCV и индикаторами
            feature_columns: Список колонок для использования в качестве признаков
            target_column: Колонка для прогнозирования (целевая переменная)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (признаки) и y (целевые значения)
        """
        # Проверяем, что все необходимые колонки присутствуют
        missing_columns = [col for col in feature_columns + [target_column] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют колонки: {missing_columns}")
            
        # Выбираем только нужные колонки
        data = df[feature_columns + [target_column]].copy()
        
        # Удаляем строки с NaN
        data.dropna(inplace=True)
        
        # Подготавливаем признаки X и целевые значения y
        X_list, y_list = [], []
        
        for i in range(len(data) - self.seq_length - self.prediction_horizon + 1):
            # Последовательность для признаков
            X_seq = data[feature_columns].iloc[i:i + self.seq_length].values
            
            # Целевое значение через prediction_horizon шагов
            y_target = data[target_column].iloc[i + self.seq_length + self.prediction_horizon - 1]
            
            X_list.append(X_seq)
            y_list.append(y_target)
        
        # Преобразуем в numpy arrays
        X = np.array(X_list)
        y = np.array(y_list).reshape(-1, 1)
        
        return X, y
    
    def normalize_data(self, X: np.ndarray, y: np.ndarray, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Нормализация данных
        
        Args:
            X: Входные признаки (numpy array)
            y: Целевые значения (numpy array)
            fit: Если True, то подгоняем скейлеры под данные, иначе используем существующие
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Нормализованные X и y
        """
        # Сохраняем оригинальную форму X
        original_shape = X.shape
        
        # Преобразуем X в 2D-массив для нормализации
        X_2d = X.reshape(-1, X.shape[-1])
        
        if fit:
            # Подгоняем скейлеры под данные
            self.feature_scaler.fit(X_2d)
            self.target_scaler.fit(y)
        
        # Нормализуем данные
        X_scaled_2d = self.feature_scaler.transform(X_2d)
        y_scaled = self.target_scaler.transform(y)
        
        # Возвращаем X к оригинальной форме
        X_scaled = X_scaled_2d.reshape(original_shape)
        
        return X_scaled, y_scaled
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Разделение данных на обучающую и тестовую выборки
        
        Args:
            X: Входные признаки (numpy array)
            y: Целевые значения (numpy array)
            test_size: Доля тестовой выборки
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        """
        # Вычисляем индекс разделения
        split_idx = int(len(X) * (1 - test_size))
        
        # Разделяем данные
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, List[float]]:
        """
        Обучение модели
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие целевые значения
            X_val: Валидационные признаки (опционально)
            y_val: Валидационные целевые значения (опционально)
            
        Returns:
            Dict[str, List[float]]: История обучения (потери на обучающей и валидационной выборках)
        """
        # Создаем датасеты
        train_dataset = PriceDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = PriceDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            use_validation = True
        else:
            use_validation = False
        
        # История обучения
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Переводим модель в режим обучения
        self.model.train()
        
        # Обучение модели
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                # Обнуляем градиенты
                self.optimizer.zero_grad()
                
                # Прямое распространение
                outputs = self.model(X_batch)
                
                # Вычисляем потери
                loss = self.criterion(outputs, y_batch)
                
                # Обратное распространение
                loss.backward()
                
                # Оптимизация
                self.optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            # Средняя потеря на эпохе
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # Валидация
            if use_validation:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self.model(X_batch)
                        loss = self.criterion(outputs, y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                
                val_loss /= len(val_loader.dataset)
                history['val_loss'].append(val_loss)
                
                self.model.train()
                
                # Логируем прогресс
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f'Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            else:
                # Логируем прогресс
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f'Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss:.4f}')
        
        self.logger.info("Обучение завершено")
        return history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Оценка модели на тестовой выборке
        
        Args:
            X_test: Тестовые признаки
            y_test: Тестовые целевые значения
            
        Returns:
            Dict[str, float]: Метрики качества модели
        """
        # Создаем датасет и загрузчик
        test_dataset = PriceDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        # Переводим модель в режим оценки
        self.model.eval()
        
        # Массивы для накопления предсказаний и реальных значений
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # Получаем предсказания
                outputs = self.model(X_batch)
                
                # Сохраняем предсказания и реальные значения
                predictions.append(outputs.cpu().numpy())
                actuals.append(y_batch.cpu().numpy())
        
        # Объединяем батчи
        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)
        
        # Денормализуем данные
        predictions_orig = self.target_scaler.inverse_transform(predictions)
        actuals_orig = self.target_scaler.inverse_transform(actuals)
        
        # Вычисляем метрики
        mse = mean_squared_error(actuals_orig, predictions_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_orig, predictions_orig)
        r2 = r2_score(actuals_orig, predictions_orig)
        
        # Вычисляем направленную точность (правильность предсказания направления движения)
        direction_actual = np.diff(actuals_orig.flatten())
        direction_pred = np.diff(predictions_orig.flatten())
        direction_accuracy = np.mean((direction_actual * direction_pred) > 0)
        
        # Собираем метрики в словарь
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }
        
        self.logger.info(f"Метрики оценки модели: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Dir.Acc={direction_accuracy:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Выполнение прогноза
        
        Args:
            X: Входные признаки (уже нормализованные)
            
        Returns:
            np.ndarray: Прогноз
        """
        # Переводим модель в режим оценки
        self.model.eval()
        
        # Создаем тензор из X
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            # Получаем предсказания
            outputs = self.model(X_tensor)
        
        # Преобразуем в numpy array
        predictions = outputs.cpu().numpy()
        
        # Денормализуем предсказания
        predictions_orig = self.target_scaler.inverse_transform(predictions)
        
        return predictions_orig
    
    def save_model(self, model_name: str = 'price_predictor'):
        """
        Сохранение модели и скейлеров
        
        Args:
            model_name: Имя для сохраняемой модели
        """
        model_path = os.path.join(self.models_dir, f'{model_name}.pth')
        scalers_path = os.path.join(self.models_dir, f'{model_name}_scalers.pkl')
        
        # Сохраняем веса модели
        torch.save(self.model.state_dict(), model_path)
        
        # Сохраняем скейлеры
        with open(scalers_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler
            }, f)
            
        self.logger.info(f"Модель и скейлеры сохранены в {self.models_dir}")
    
    def load_model(self, model_name: str = 'price_predictor'):
        """
        Загрузка модели и скейлеров
        
        Args:
            model_name: Имя загружаемой модели
        """
        model_path = os.path.join(self.models_dir, f'{model_name}.pth')
        scalers_path = os.path.join(self.models_dir, f'{model_name}_scalers.pkl')
        
        # Проверяем существование файлов
        if not os.path.exists(model_path) or not os.path.exists(scalers_path):
            self.logger.error(f"Не найдены файлы модели {model_name}")
            return False
        
        try:
            # Загружаем веса модели
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            
            # Загружаем скейлеры
            with open(scalers_path, 'rb') as f:
                scalers = pickle.load(f)
                self.feature_scaler = scalers['feature_scaler']
                self.target_scaler = scalers['target_scaler']
                
            self.logger.info(f"Модель и скейлеры загружены из {self.models_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def plot_results(self, predictions: np.ndarray, actuals: np.ndarray, title: str = 'Сравнение прогноза и реальных цен'):
        """
        Визуализация результатов прогнозирования
        
        Args:
            predictions: Прогнозы модели
            actuals: Реальные значения
            title: Заголовок графика
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actuals, label='Реальные цены', color='blue')
        plt.plot(predictions, label='Прогноз', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Время')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True)
        plt.show()

# Функция для тестирования модели на искусственных данных
def test_model_on_synthetic_data():
    """
    Тестирование модели на синтетических данных
    """
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ ПРОГНОЗИРОВАНИЯ ЦЕНЫ")
    print("=" * 50)
    
    # Создаем экземпляр модели
    predictor = PricePredictor()
    
    # Создаем синтетические данные
    # Синусоида с шумом для имитации цен
    n_samples = 1000
    time = np.arange(n_samples)
    price = 100 + 10 * np.sin(2 * np.pi * time / 100) + np.random.normal(0, 1, n_samples)
    
    # Создаем другие признаки
    features = []
    for i in range(predictor.input_size - 1):  # -1, так как одним из признаков будет сама цена
        # Зашумленный синус с разной частотой
        feature = 50 + 5 * np.sin(2 * np.pi * time / (50 + i * 10)) + np.random.normal(0, 0.5, n_samples)
        features.append(feature)
    
    # Собираем все в DataFrame
    columns = ['price'] + [f'feature_{i}' for i in range(predictor.input_size - 1)]
    data = np.column_stack([price] + features)
    df = pd.DataFrame(data, columns=columns)
    
    print("Создан синтетический датасет для тестирования")
    print(f"Размер данных: {df.shape}")
    print(df.head())
    
    # Подготавливаем данные для обучения
    feature_columns = [f'feature_{i}' for i in range(predictor.input_size - 1)] + ['price']
    X, y = predictor.prepare_data(df, feature_columns, target_column='price')
    
    print(f"Подготовлены данные: X={X.shape}, y={y.shape}")
    
    # Нормализуем данные
    X_scaled, y_scaled = predictor.normalize_data(X, y)
    
    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = predictor.train_test_split(X_scaled, y_scaled, test_size=0.2)
    
    print(f"Обучающая выборка: X={X_train.shape}, y={y_train.shape}")
    print(f"Тестовая выборка: X={X_test.shape}, y={y_test.shape}")
    
    # Обучаем модель с меньшим числом эпох для быстроты
    predictor.num_epochs = 20
    
    # Если размерность не соответствует, корректируем модель
    if X_train.shape[2] != predictor.input_size:
        print(f"Корректировка размерности модели: {X_train.shape[2]} вместо {predictor.input_size}")
        # Пересоздаем модель с правильной размерностью
        predictor.input_size = X_train.shape[2]
        predictor.model = LSTM_PricePredictor(
            input_size=predictor.input_size,
            hidden_size=predictor.hidden_size,
            num_layers=predictor.num_layers,
            output_size=predictor.output_size,
            dropout=predictor.dropout
        ).to(DEVICE)
        # Пересоздаем оптимизатор
        predictor.optimizer = optim.Adam(predictor.model.parameters(), lr=predictor.learning_rate)
    
    history = predictor.train(X_train, y_train, X_test, y_test)
    
    # Оцениваем модель
    metrics = predictor.evaluate(X_test, y_test)
    
    # Делаем прогноз
    predictions = predictor.predict(X_test)
    actuals = predictor.target_scaler.inverse_transform(y_test)
    
    # Сохраняем модель
    predictor.save_model('synthetic_test')
    
    # Визуализируем результаты (первые 100 точек)
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:100].flatten(), label='Реальные цены', color='blue')
    plt.plot(predictions[:100].flatten(), label='Прогноз', color='red', linestyle='--')
    plt.title('Сравнение прогноза и реальных цен')
    plt.xlabel('Время')
    plt.ylabel('Цена')
    plt.legend()
    plt.grid(True)
    
    # Сохраняем график
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'price_prediction_test.png'))
    
    print("\nТест завершен. График сохранен в директории data/plots")
    print("=" * 50)

if __name__ == "__main__":
    # Запускаем тест на синтетических данных
    test_model_on_synthetic_data() 