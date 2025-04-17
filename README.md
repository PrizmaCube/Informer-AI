# Система автоматического резервного копирования проекта

Этот инструмент позволяет легко создавать резервные копии проекта, помечать важные версии тегами и восстанавливать предыдущие состояния при необходимости.

## О проекте

Проект предназначен для торговли на бирже OKX через WebSocket API и автоматически сохраняет все изменения с помощью системы контроля версий Git.

## Установка

```bash
# Уже выполнено: инициализация Git-репозитория
git init

# Уже выполнено: создание первоначальной резервной копии
git add .
git commit -m "Первоначальная версия проекта"
```

## Использование скрипта auto_backup.py

### Создание резервной копии

```bash
# Простое резервное копирование
python auto_backup.py backup

# Резервное копирование с сообщением
python auto_backup.py backup -m "Описание изменений"

# Создание именованной версии (тега)
python auto_backup.py backup -t --tag-name v1.0 --tag-message "Стабильная версия"

# Самый простой вариант, без дополнительных аргументов
python auto_backup.py
```

### Просмотр доступных версий

```bash
python auto_backup.py list
```

### Восстановление предыдущей версии

```bash
python auto_backup.py rollback v1.0
```

## Ручные Git-команды

### Ежедневные операции

```bash
# Проверка статуса изменений
git status

# Создание резервной копии
git add .
git commit -m "Описание изменений"

# Добавление тега (версии)
git tag -a v1.0 -m "Описание версии"
```

### Работа с версиями

```bash
# Просмотр списка версий
git tag -l

# Просмотр деталей версии
git show v1.0

# Восстановление к версии
git checkout v1.0

# Возврат к последней версии
git checkout master
```

### Просмотр истории

```bash
# Просмотр истории изменений
git log

# Просмотр графика изменений
git log --graph --oneline --all
```

## Автоматизация резервного копирования

Для автоматического создания резервных копий через определенные интервалы вы можете:

- **Windows**: Использовать планировщик задач и добавить запуск `python auto_backup.py` с нужной периодичностью
- **Linux/Mac**: Настроить cron-задачу для запуска скрипта

## Рекомендации

- Создавайте резервные копии (коммиты) как минимум в конце каждого рабочего дня
- Используйте теги (версии) для маркировки важных этапов разработки
- При сохранении резервной копии указывайте понятное описание внесенных изменений 