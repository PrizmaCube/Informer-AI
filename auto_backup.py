#!/usr/bin/env python
import os
import sys
import subprocess
import datetime
import argparse

def run_command(command):
    """Выполнение команды и вывод результата"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               encoding='utf-8')
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Ошибка: {e}")
        print(f"Текст ошибки: {e.stderr}")
        sys.exit(1)

def create_backup(message=None):
    """Создание резервной копии (коммита) текущего состояния проекта"""
    # Проверяем изменённые файлы
    changes = run_command("git status --porcelain")
    
    if not changes:
        print("Нет изменений для сохранения.")
        return False
    
    # Добавляем все изменения в индекс
    run_command("git add .")
    
    # Формируем сообщение коммита
    if not message:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Автоматическое резервное копирование: {timestamp}"
    
    # Создаём коммит
    run_command(f'git commit -m "{message}"')
    
    print(f"Резервная копия создана с сообщением: {message}")
    return True

def create_tag(tag_name=None, message=None):
    """Создание тега для текущей версии проекта"""
    if not tag_name:
        timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
        tag_name = f"v{timestamp}"
    
    if not message:
        message = f"Версия {tag_name}"
    
    run_command(f'git tag -a {tag_name} -m "{message}"')
    print(f"Создан тег: {tag_name}")
    return tag_name

def list_tags():
    """Вывод списка всех тегов (версий)"""
    tags = run_command("git tag -l --sort=-creatordate")
    if not tags:
        print("Нет сохранённых версий.")
        return
    
    print("Доступные версии:")
    for tag in tags.split('\n'):
        tag_info = run_command(f'git show {tag} --pretty=format:"%ai %s" -s')
        print(f"  {tag} - {tag_info}")

def rollback_to_tag(tag_name):
    """Возврат к указанной версии (тегу)"""
    # Проверяем, существует ли тег
    tags = run_command("git tag -l").split('\n')
    if tag_name not in tags:
        print(f"Ошибка: Версия {tag_name} не найдена.")
        return False
    
    # Проверяем наличие несохраненных изменений
    changes = run_command("git status --porcelain")
    if changes:
        save = input("Есть несохраненные изменения. Сохранить их перед откатом? (д/н): ")
        if save.lower() in ('д', 'y', 'yes', 'да'):
            create_backup()
    
    # Выполняем откат
    run_command(f"git checkout {tag_name}")
    print(f"Выполнен откат к версии {tag_name}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Автоматическое резервное копирование проекта")
    subparsers = parser.add_subparsers(dest="command", help="Команда")
    
    # Команда backup
    backup_parser = subparsers.add_parser("backup", help="Создать резервную копию")
    backup_parser.add_argument("-m", "--message", help="Сообщение для коммита")
    backup_parser.add_argument("-t", "--tag", action="store_true", help="Создать тег для версии")
    backup_parser.add_argument("--tag-name", help="Имя тега")
    backup_parser.add_argument("--tag-message", help="Описание тега")
    
    # Команда list
    subparsers.add_parser("list", help="Показать список версий")
    
    # Команда rollback
    rollback_parser = subparsers.add_parser("rollback", help="Откатиться к указанной версии")
    rollback_parser.add_argument("tag", help="Имя тега/версии для отката")
    
    args = parser.parse_args()
    
    if not args.command or args.command == "backup":
        # Создаём резервную копию
        success = create_backup(args.message if hasattr(args, 'message') else None)
        
        # Если запрошено и резервная копия создана, добавляем тег
        if success and hasattr(args, 'tag') and args.tag:
            create_tag(
                args.tag_name if hasattr(args, 'tag_name') else None,
                args.tag_message if hasattr(args, 'tag_message') else None
            )
    
    elif args.command == "list":
        list_tags()
    
    elif args.command == "rollback":
        rollback_to_tag(args.tag)

if __name__ == "__main__":
    main() 