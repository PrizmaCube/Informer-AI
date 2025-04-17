#!/usr/bin/env python
import os
import subprocess
import requests
import zipfile
import shutil
import re

def run_command(cmd):
    """Выполнить команду и вернуть результат"""
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return process.stdout.strip()

def get_versions():
    """Получить список версий с GitHub"""
    # Получаем имя пользователя и репозитория из настроек Git
    remote_url = run_command("git remote get-url origin")
    match = re.search(r'github\.com[:/]([^/]+)/([^/.]+)', remote_url)
    
    if not match:
        print("Ошибка: Не удалось определить репозиторий GitHub")
        return []
        
    username, repo = match.group(1), match.group(2)
    
    # Получаем список тегов с GitHub
    api_url = f"https://api.github.com/repos/{username}/{repo}/tags"
    response = requests.get(api_url)
    
    if response.status_code != 200:
        print(f"Ошибка получения версий: {response.status_code}")
        return []
        
    tags = response.json()
    return [(tag["name"], username, repo) for tag in tags]

def restore_version(version_info=None):
    """Восстановить проект к выбранной версии"""
    # Если версия не указана, показываем список доступных версий
    if not version_info:
        versions = get_versions()
        if not versions:
            print("Нет доступных версий")
            return False
            
        print("\nДоступные версии:")
        for i, (version, _, _) in enumerate(versions):
            print(f"{i+1}. {version}")
            
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
    
    version, username, repo = version_info
    
    # Спрашиваем подтверждение
    print(f"\nВНИМАНИЕ: Это действие заменит ВСЕ файлы в текущем проекте файлами из версии {version}!")
    confirm = input("Все несохраненные изменения будут потеряны. Продолжить? (д/н): ")
    
    if confirm.lower() not in ["д", "y", "yes", "да"]:
        print("Операция отменена")
        return False
    
    print(f"Загрузка версии {version}...")
    
    # Загружаем архив с GitHub
    zip_url = f"https://github.com/{username}/{repo}/archive/refs/tags/{version}.zip"
    temp_zip = "temp_version.zip"
    
    try:
        # Загрузка архива
        response = requests.get(zip_url, stream=True)
        with open(temp_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Определяем путь для временной распаковки
        temp_dir = f"temp_{repo}_{version}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        # Распаковываем архив
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
        
        # Сохраняем .git папку
        git_dir = os.path.join(os.getcwd(), ".git")
        temp_git_dir = None
        
        if os.path.exists(git_dir):
            temp_git_dir = os.path.join(os.getcwd(), "temp_git_backup")
            if os.path.exists(temp_git_dir):
                shutil.rmtree(temp_git_dir)
            shutil.copytree(git_dir, temp_git_dir)
        
        # Удаляем все файлы из текущего каталога кроме временных и скрипта
        current_script = os.path.basename(__file__)
        for item in os.listdir(os.getcwd()):
            item_path = os.path.join(os.getcwd(), item)
            if item not in [temp_dir, temp_zip, current_script, "temp_git_backup"]:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        # Копируем все файлы из архива
        for item in os.listdir(extracted_root):
            src_path = os.path.join(extracted_root, item)
            dst_path = os.path.join(os.getcwd(), item)
            
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        
        # Восстанавливаем .git папку
        if temp_git_dir:
            shutil.rmtree(git_dir, ignore_errors=True)
            shutil.copytree(temp_git_dir, git_dir)
            shutil.rmtree(temp_git_dir)
            
        # Копируем текущий скрипт обратно, если он был перезаписан
        if not os.path.exists(current_script):
            shutil.copy2(__file__, current_script)
            
        # Очистка
        os.remove(temp_zip)
        shutil.rmtree(temp_dir)
        
        print(f"\nУспешно! Проект восстановлен к версии {version}")
        return True
        
    except Exception as e:
        print(f"Ошибка при восстановлении версии: {e}")
        return False

if __name__ == "__main__":
    print("=== Инструмент полного восстановления проекта к выбранной версии ===\n")
    restore_version() 