#!/usr/bin/env python
import os
import sys
import subprocess
import requests
import zipfile
import shutil
import re
import argparse
import json
from datetime import datetime

def run_command(cmd):
    """Выполнить команду и вернуть результат"""
    try:
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return process.stdout.strip()
    except Exception as e:
        print(f"Предупреждение: ошибка выполнения команды: {e}")
        return ""

def get_versions(username=None, repo=None):
    """Получить список версий с GitHub с дополнительной информацией"""
    # Если репозиторий не указан, пытаемся определить из настроек Git
    if not username or not repo:
        # Получаем имя пользователя и репозитория из настроек Git
        remote_url = run_command("git remote get-url origin")
        match = re.search(r'github\.com[:/]([^/]+)/([^/.]+)', remote_url)
        
        if not match:
            print("Ошибка: Не удалось определить репозиторий GitHub")
            print("Убедитесь, что вы находитесь в Git репозитории и настроена связь с GitHub")
            print("Или укажите репозиторий вручную с помощью опций --username и --repo")
            return []
            
        username, repo = match.group(1), match.group(2)
        
    print(f"Получение версий из репозитория GitHub: {username}/{repo}")
    
    # Получаем список тегов с GitHub
    api_url = f"https://api.github.com/repos/{username}/{repo}/tags"
    print(f"Запрос данных с GitHub API: {api_url}")
    
    try:
        response = requests.get(api_url)
        
        if response.status_code != 200:
            print(f"Ошибка получения версий: {response.status_code}")
            print(f"Ответ API: {response.text}")
            return []
            
        tags = response.json()
        
        # Получаем дополнительную информацию для каждого тега
        versions = []
        for tag in tags:
            # Получаем информацию о коммите для этого тега
            commit_url = tag.get('commit', {}).get('url')
            if commit_url:
                try:
                    commit_response = requests.get(commit_url)
                    if commit_response.status_code == 200:
                        commit_data = commit_response.json()
                        
                        # Получаем дату коммита и сообщение
                        commit_date = commit_data.get('commit', {}).get('committer', {}).get('date', '')
                        if commit_date:
                            try:
                                # Преобразуем формат даты GitHub в читаемый формат
                                dt = datetime.strptime(commit_date, '%Y-%m-%dT%H:%M:%SZ')
                                date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                date_str = commit_date
                        else:
                            date_str = "Неизвестно"
                            
                        # Получаем сообщение коммита
                        message = commit_data.get('commit', {}).get('message', '')
                        # Берем только первую строку сообщения
                        message = message.split('\n')[0] if message else "Нет описания"
                        
                        versions.append((tag["name"], username, repo, date_str, message))
                        continue
                except Exception as e:
                    print(f"Ошибка при получении данных коммита: {e}")
            
            # Если не удалось получить дополнительную информацию, добавляем только базовую
            versions.append((tag["name"], username, repo, "Неизвестно", "Нет описания"))
            
        return versions
    except Exception as e:
        print(f"Ошибка при получении версий: {e}")
        return []

def get_latest_version(username=None, repo=None):
    """Получить последнюю версию из GitHub"""
    versions = get_versions(username, repo)
    if not versions:
        return None
    return versions[0]  # Первая версия в списке - самая последняя

def restore_version(version_info=None, force=False, latest=False, keep_git=False, 
                   username=None, repo=None, no_restore_scripts=False):
    """Восстановить проект к выбранной версии"""
    # Если запрошена последняя версия, находим её
    if latest:
        latest_version = get_latest_version(username, repo)
        if not latest_version:
            print("Нет доступных версий на GitHub")
            if username and repo:
                print(f"Проверьте правильность указанного репозитория: {username}/{repo}")
            return False
        version_info = latest_version
        print(f"Выбрана последняя версия: {latest_version[0]}")
    
    # Если версия не указана, показываем список доступных версий
    if not version_info:
        versions = get_versions(username, repo)
        if not versions:
            print("Нет доступных версий")
            return False
            
        print("\nДоступные версии:")
        print("=================")
        for i, (version, _, _, date, message) in enumerate(versions):
            print(f"{i+1}. {version} ({date}) - {message}")
            
        choice = input("\nВыберите версию (номер или название): ")
        try:
            if choice.isdigit() and 1 <= int(choice) <= len(versions):
                version_info = versions[int(choice) - 1]
            else:
                # Ищем версию по имени
                for v in versions:
                    if v[0] == choice:
                        version_info = v
                        break
                else:
                    print(f"Версия '{choice}' не найдена")
                    return False
        except:
            print("Неверный выбор")
            return False
    
    # Извлекаем основную информацию о версии
    if isinstance(version_info, str):
        # Если передана строка с версией, создаем инфо с заданным username и repo
        if not username or not repo:
            print("Ошибка: при указании версии строкой нужно также указать --username и --repo")
            return False
        version = version_info
        print(f"Выбрана версия: {version}")
    elif len(version_info) >= 5:  # Если версия содержит дополнительные данные
        version, username, repo, date, message = version_info
        print(f"Выбрана версия: {version} ({date}) - {message}")
    else:  # Если версия содержит только базовую информацию
        version, username, repo = version_info[:3]
        print(f"Выбрана версия: {version}")
    
    # Спрашиваем подтверждение, если не указан флаг force
    if not force:
        print(f"\nВНИМАНИЕ: Это действие заменит ВСЕ файлы в текущем проекте файлами из версии {version}!")
        confirm = input("Все несохраненные изменения будут потеряны. Продолжить? (д/н): ")
        
        if confirm.lower() not in ["д", "y", "yes", "да"]:
            print("Операция отменена")
            return False
    
    print(f"Загрузка версии {version}...")
    
    # Загружаем архив с GitHub
    zip_url = f"https://github.com/{username}/{repo}/archive/refs/tags/{version}.zip"
    print(f"Загрузка архива: {zip_url}")
    temp_zip = "temp_version.zip"
    
    try:
        # Загрузка архива
        print(f"Скачивание архива с GitHub...")
        response = requests.get(zip_url, stream=True)
        if response.status_code != 200:
            print(f"Ошибка загрузки архива: HTTP {response.status_code}")
            print(f"Ответ сервера: {response.text[:200]}")
            return False
            
        with open(temp_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Архив загружен. Размер: {os.path.getsize(temp_zip) / 1024:.1f} КБ")
        
        # Определяем путь для временной распаковки
        temp_dir = f"temp_{repo}_{version}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir)
        
        # Распаковываем архив
        print(f"Распаковка архива в: {temp_dir}")
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Определяем корневую папку в распакованном архиве
        extracted_root = None
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path):
                extracted_root = item_path
                break
        
        if not extracted_root:
            print("Ошибка: не удалось найти корневую папку в архиве")
            return False
        
        print(f"Найдена корневая папка архива: {os.path.basename(extracted_root)}")
        
        # Сохраняем .git папку
        git_dir = os.path.join(os.getcwd(), ".git")
        temp_git_dir = None
        
        if os.path.exists(git_dir):
            if keep_git:
                print("Сохранение Git репозитория без копирования...")
                # В этом режиме мы не копируем .git, а просто помечаем для сохранения
            else:
                print("Сохранение Git репозитория...")
                try:
                    temp_git_dir = os.path.join(os.getcwd(), "temp_git_backup")
                    if os.path.exists(temp_git_dir):
                        shutil.rmtree(temp_git_dir, ignore_errors=True)
                    
                    # Используем более безопасный метод копирования
                    os.makedirs(temp_git_dir)
                    
                    # Копируем файлы с обработкой ошибок
                    for root, dirs, files in os.walk(git_dir):
                        rel_dir = os.path.relpath(root, git_dir)
                        if rel_dir == '.':
                            rel_dir = ''
                        
                        # Создаем поддиректории
                        if rel_dir:
                            os.makedirs(os.path.join(temp_git_dir, rel_dir), exist_ok=True)
                        
                        # Копируем файлы
                        for file in files:
                            src_file = os.path.join(root, file)
                            dst_file = os.path.join(temp_git_dir, rel_dir, file)
                            try:
                                shutil.copy2(src_file, dst_file)
                            except (PermissionError, OSError) as e:
                                print(f"Предупреждение: не удалось скопировать Git файл: {src_file}")
                                print(f"  Причина: {str(e)}")
                                print("  Продолжаем работу...")
                except Exception as e:
                    print(f"Предупреждение: проблема с резервным копированием Git репозитория: {e}")
                    print("Продолжение без резервной копии Git...")
                    temp_git_dir = None
        
        # Удаляем все файлы из текущего каталога кроме временных и скрипта
        print("Удаление текущих файлов проекта...")
        current_script = os.path.basename(__file__)
        # Также сохраним другие скрипты Git-управления, если они есть
        git_scripts = ["git_backup.py", "git_download.py", "git_restore.py", 
                        "git_restore_readme.md", "git_download_readme.md", "Инструкция отката",
                        "Инструкция сохранения", "Инструкция git_download"]
        
        # Сохраняем содержимое скриптов
        script_backup = {}
        if not no_restore_scripts:
            for script in git_scripts:
                if os.path.exists(script):
                    try:
                        with open(script, 'rb') as f:
                            script_backup[script] = f.read()
                    except Exception as e:
                        print(f"Предупреждение: не удалось сохранить скрипт {script}: {e}")
        
        # Удаляем файлы с обработкой ошибок
        for item in os.listdir(os.getcwd()):
            item_path = os.path.join(os.getcwd(), item)
            if item in [temp_dir, temp_zip, "temp_git_backup"]:
                continue
            
            # Если используется опция keep_git и это папка .git, пропускаем
            if keep_git and item == ".git":
                continue
                
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                else:
                    os.remove(item_path)
            except (PermissionError, OSError) as e:
                print(f"Предупреждение: не удалось удалить {item_path}: {e}")
                print("  Пропускаем этот файл и продолжаем...")
        
        # Копируем все файлы из архива
        print("Копирование файлов из выбранной версии...")
        file_count = 0
        
        # Копируем файлы с обработкой ошибок
        for root, dirs, files in os.walk(extracted_root):
            rel_dir = os.path.relpath(root, extracted_root)
            if rel_dir == '.':
                rel_dir = ''
                
            # Создаем поддиректории
            if rel_dir:
                try:
                    os.makedirs(os.path.join(os.getcwd(), rel_dir), exist_ok=True)
                except Exception as e:
                    print(f"Ошибка создания директории {rel_dir}: {e}")
                    continue
                    
            # Копируем файлы
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(os.getcwd(), rel_dir, file) if rel_dir else os.path.join(os.getcwd(), file)
                try:
                    shutil.copy2(src_file, dst_file)
                    file_count += 1
                except Exception as e:
                    print(f"Ошибка копирования файла {file}: {e}")
        
        # Восстанавливаем .git папку
        if temp_git_dir and not keep_git:
            print("Восстановление Git репозитория...")
            try:
                if os.path.exists(git_dir):
                    shutil.rmtree(git_dir, ignore_errors=True)
                
                # Используем более безопасный метод копирования
                os.makedirs(git_dir, exist_ok=True)
                
                # Копируем файлы с обработкой ошибок
                for root, dirs, files in os.walk(temp_git_dir):
                    rel_dir = os.path.relpath(root, temp_git_dir)
                    if rel_dir == '.':
                        rel_dir = ''
                    
                    # Создаем поддиректории
                    if rel_dir:
                        os.makedirs(os.path.join(git_dir, rel_dir), exist_ok=True)
                    
                    # Копируем файлы
                    for file in files:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(git_dir, rel_dir, file)
                        try:
                            shutil.copy2(src_file, dst_file)
                        except (PermissionError, OSError) as e:
                            print(f"Предупреждение: не удалось восстановить Git файл: {dst_file}")
                            print(f"  Причина: {str(e)}")
                
                # Удаляем временную копию
                try:
                    shutil.rmtree(temp_git_dir)
                except Exception as e:
                    print(f"Предупреждение: не удалось удалить временную копию Git: {e}")
                    
            except Exception as e:
                print(f"Ошибка при восстановлении Git репозитория: {e}")
                print("Git репозиторий может быть поврежден. Рекомендуется клонировать репозиторий заново.")
            
        # Восстанавливаем скрипты Git-управления
        if not no_restore_scripts and script_backup:
            print("Восстановление скриптов Git-управления...")
            for script, content in script_backup.items():
                try:
                    with open(script, 'wb') as f:
                        f.write(content)
                except Exception as e:
                    print(f"Ошибка восстановления скрипта {script}: {e}")
            
        # Очистка
        try:
            os.remove(temp_zip)
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Предупреждение: не удалось выполнить очистку временных файлов: {e}")
        
        print(f"\nУспешно! Проект восстановлен к версии {version}")
        print(f"Скопировано файлов: {file_count}")
        return True
        
    except Exception as e:
        print(f"Ошибка при восстановлении версии: {e}")
        return False

def main():
    """Основная функция с обработкой параметров командной строки"""
    parser = argparse.ArgumentParser(description="Инструмент для полного восстановления проекта к выбранной версии с GitHub")
    parser.add_argument("version", nargs="?", help="Версия для восстановления (опционально)")
    parser.add_argument("-f", "--force", action="store_true", help="Принудительное восстановление без запроса подтверждения")
    parser.add_argument("-l", "--list", action="store_true", help="Только вывести список доступных версий")
    parser.add_argument("--latest", action="store_true", help="Быстрый откат к последней доступной версии")
    parser.add_argument("--temp", action="store_true", help="Временное восстановление последней версии (то же что и --latest)")
    parser.add_argument("--keep-git", action="store_true", help="Сохранить .git директорию без изменений (не копировать)")
    parser.add_argument("--admin", action="store_true", help="Запросить повышение прав администратора (только Windows)")
    parser.add_argument("--username", help="Имя пользователя GitHub (если не настроен Git)")
    parser.add_argument("--repo", help="Имя репозитория GitHub (если не настроен Git)")
    parser.add_argument("--no-restore-scripts", action="store_true", help="Не восстанавливать скрипты управления версиями")
    
    args = parser.parse_args()
    
    # Проверяем наличие обоих параметров репозитория
    if (args.username and not args.repo) or (args.repo and not args.username):
        print("Ошибка: необходимо указать оба параметра --username и --repo")
        return
    
    if args.list:
        # Просто выводим список версий без восстановления
        versions = get_versions(args.username, args.repo)
        if versions:
            print("\nДоступные версии:")
            print("=================")
            for i, (version, _, _, date, message) in enumerate(versions):
                print(f"{i+1}. {version} ({date}) - {message}")
        return
    
    # Проверка необходимости прав администратора
    if args.admin and os.name == 'nt':
        try:
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                print("Запрос прав администратора...")
                # Перезапускаем скрипт с правами администратора
                script = os.path.abspath(__file__)
                params = ' '.join(sys.argv[1:])
                # Удаляем флаг admin, чтобы избежать рекурсии
                params = params.replace('--admin', '')
                ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
                sys.exit(0)
            else:
                print("Скрипт запущен с правами администратора")
        except Exception as e:
            print(f"Ошибка при запросе прав администратора: {e}")
            print("Продолжаем без повышения прав...")
    
    # Обработка флагов быстрого отката к последней версии
    if args.latest or args.temp:
        restore_version(
            force=args.force, 
            latest=True, 
            keep_git=args.keep_git, 
            username=args.username, 
            repo=args.repo,
            no_restore_scripts=args.no_restore_scripts
        )
        return
        
    if args.version:
        if args.username and args.repo:
            # Если указан username и repo, используем прямое восстановление
            restore_version(
                args.version, 
                args.force, 
                keep_git=args.keep_git, 
                username=args.username, 
                repo=args.repo,
                no_restore_scripts=args.no_restore_scripts
            )
        else:
            # Попытка получить информацию о конкретной версии
            versions = get_versions()
            version_info = None
            for v in versions:
                if v[0] == args.version:
                    version_info = v
                    break
                    
            if not version_info:
                print(f"Версия '{args.version}' не найдена")
                return
                
            restore_version(
                version_info, 
                args.force, 
                keep_git=args.keep_git,
                no_restore_scripts=args.no_restore_scripts
            )
    else:
        # Интерактивный выбор версии
        restore_version(
            force=args.force, 
            keep_git=args.keep_git,
            username=args.username, 
            repo=args.repo,
            no_restore_scripts=args.no_restore_scripts
        )

if __name__ == "__main__":
    print("=== Инструмент полного восстановления проекта к выбранной версии ===\n")
    main() 