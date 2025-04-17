#!/usr/bin/env python
import os
import sys
import subprocess
import datetime
import argparse
import re

def run_command(command, show_output=False):
    """Execute command and return output"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               encoding='utf-8')
        if show_output:
            print(result.stdout)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Error message: {e.stderr}")
        sys.exit(1)

def setup_github_remote():
    """Setup GitHub remote repository if not already configured"""
    try:
        remotes = run_command("git remote -v")
        if "origin" not in remotes:
            repo_url = input("Enter your GitHub repository URL (https://github.com/username/repo.git): ")
            if repo_url:
                run_command(f"git remote add origin {repo_url}")
                print(f"GitHub repository {repo_url} configured as remote 'origin'")
        return True
    except Exception as e:
        print(f"Error setting up GitHub remote: {e}")
        return False

def get_latest_version():
    """Get the latest version tag from git"""
    try:
        tags = run_command("git tag -l 'v*' --sort=-v:refname").split('\n')
        if not tags or not tags[0]:
            return "v0.0"
        
        return tags[0]
    except Exception:
        return "v0.0"

def tag_exists(tag):
    """Check if a tag exists"""
    try:
        tags = run_command("git tag -l").split('\n')
        return tag in tags
    except Exception:
        return False

def increment_version(version):
    """Increment the version number"""
    # Extract version components
    match = re.match(r'v(\d+)\.(\d+)', version)
    if not match:
        return "v1.0"
    
    major, minor = map(int, match.groups())
    # Increment minor version
    return f"v{major}.{minor+1}"

def find_next_available_version(version):
    """Find the next available version that doesn't already exist as a tag"""
    new_version = version
    while tag_exists(new_version):
        new_version = increment_version(new_version)
    return new_version

def save_version(message=None, push=False, auto_increment=True, force=False):
    """Save a new version with auto-incrementing version number"""
    # Check for changes
    changes = run_command("git status --porcelain")
    if not changes and not force:
        print("No changes to save.")
        user_choice = input("Хотите создать новую версию без изменений? (д/н): ")
        if user_choice.lower() not in ["д", "y", "yes", "да"]:
            return False
        force = True
    
    # Get current and new version
    current_version = get_latest_version()
    if auto_increment:
        new_version = increment_version(current_version)
        # Make sure the new version doesn't already exist
        new_version = find_next_available_version(new_version)
    else:
        new_version = input(f"Enter new version (current: {current_version}): ")
        if not new_version:
            new_version = increment_version(current_version)
            new_version = find_next_available_version(new_version)
        elif tag_exists(new_version):
            print(f"Версия {new_version} уже существует.")
            overwrite = input("Перезаписать существующую версию? (д/н): ")
            if overwrite.lower() in ["д", "y", "yes", "да"]:
                # Delete existing tag
                run_command(f"git tag -d {new_version}")
                if push:
                    # If we're going to push, also delete the remote tag
                    try:
                        run_command(f"git push origin :refs/tags/{new_version}")
                        print(f"Удалена удаленная версия {new_version}")
                    except Exception:
                        print(f"Не удалось удалить удаленную версию {new_version}. Продолжаем...")
                print(f"Существующая версия {new_version} удалена.")
            else:
                new_version = find_next_available_version(new_version)
                print(f"Будет создана новая версия {new_version} вместо.")
    
    # Get description if not provided
    if not message:
        message = input("Enter version description: ")
        if not message:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"Update {timestamp}"
    
    # Add all changes
    run_command("git add .")
    
    # Create commit
    commit_message = f'Version {new_version}: {message}'
    if force and not changes:
        commit_message += ' [empty update]'
    
    # Use --allow-empty if forcing an empty commit
    commit_cmd = 'git commit -m "{}"'.format(commit_message)
    if force and not changes:
        commit_cmd = 'git commit --allow-empty -m "{}"'.format(commit_message)
    
    run_command(commit_cmd)
    
    # Create tag
    try:
        run_command(f'git tag -a {new_version} -m "{message}"')
        print(f"Version {new_version} saved: {message}")
    except Exception as e:
        print(f"Error creating tag: {e}")
        return False
    
    # Push to GitHub if requested
    if push:
        print("Pushing to GitHub...")
        try:
            run_command("git push origin master")
            run_command("git push origin --tags")
            print("Successfully pushed to GitHub")
        except Exception as e:
            print(f"Error pushing to GitHub: {e}")
            return True
    
    return True

def list_versions():
    """List all versions with details"""
    tags = run_command("git tag -l 'v*' --sort=-v:refname")
    if not tags:
        print("No saved versions.")
        return
    
    print("\nAvailable versions:")
    print("===================")
    
    for tag in tags.split('\n'):
        if not tag:
            continue
            
        # Get tag date and message
        info = run_command(f'git show {tag} --pretty=format:"%ci | %s" -s')
        date, message = info.split(" | ", 1) if " | " in info else (info, "")
        
        # Get number of commits in this version
        if tag == tags.split('\n')[0]:  # If this is the latest version
            commit_count = run_command(f'git rev-list --count {tag}')
        else:
            next_tag_index = tags.split('\n').index(tag) - 1
            next_tag = tags.split('\n')[next_tag_index]
            commit_count = run_command(f'git rev-list --count {next_tag}..{tag}')
        
        print(f"{tag} ({date}) - {message}")
        print(f"  Changes: {commit_count} commits")
        # Show the main files that changed in this version
        if tag == tags.split('\n')[0]:  # If this is the latest version
            files = run_command(f'git show {tag} --name-only --pretty=format:""').strip()
        else:
            next_tag_index = tags.split('\n').index(tag) - 1
            next_tag = tags.split('\n')[next_tag_index]
            files = run_command(f'git diff --name-only {next_tag} {tag}').strip()
        
        if files:
            print("  Modified files:")
            file_list = files.split('\n')
            for file in file_list[:5]:  # Show max 5 files
                if file:
                    print(f"    - {file}")
            if len(file_list) > 5:
                print(f"    - ... and {len(file_list) - 5} more files")
        print()

def rollback(version):
    """Rollback to a specific version"""
    # Check if version exists
    tags = run_command("git tag -l").split('\n')
    if version not in tags:
        print(f"Error: Version {version} not found.")
        return False
    
    # Check for uncommitted changes
    changes = run_command("git status --porcelain")
    if changes:
        save = input("There are unsaved changes. Save before rollback? (y/n): ")
        if save.lower() == 'y':
            save_version()
    
    # Do rollback
    run_command(f"git checkout {version}")
    print(f"Rolled back to version {version}")
    
    # Warn about detached head
    print("\nNOTE: You are now in 'detached HEAD' state.")
    print("To return to the latest version, use: python git_backup.py latest")
    return True

def return_to_latest():
    """Return to the latest version (master branch)"""
    run_command("git checkout master")
    print("Returned to the latest version")
    return True

def initialize():
    """Initialize a new Git project"""
    # Check if .git exists
    if os.path.exists(".git"):
        print("Git repository already initialized.")
        return
    
    # Initialize git
    run_command("git init")
    
    # Create .gitignore if it doesn't exist
    if not os.path.exists(".gitignore"):
        with open(".gitignore", "w") as f:
            f.write("""# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
""")
        print("Created .gitignore file")
    
    # Add files and make initial commit
    run_command("git add .")
    run_command('git commit -m "Initial project setup"')
    run_command('git tag -a v0.1 -m "Initial version"')
    
    print("Git repository initialized with initial version v0.1")
    
    # Setup GitHub remote
    setup_github_remote()

def main():
    parser = argparse.ArgumentParser(description="Easy version control with GitHub integration")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # save command
    save_parser = subparsers.add_parser("save", help="Save a new version")
    save_parser.add_argument("-m", "--message", help="Version description")
    save_parser.add_argument("-p", "--push", action="store_true", help="Push to GitHub")
    save_parser.add_argument("--manual", action="store_true", help="Manually enter version number")
    save_parser.add_argument("-f", "--force", action="store_true", help="Force creating a new version even without changes")
    
    # list command
    subparsers.add_parser("list", help="List all versions")
    
    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to a specific version")
    rollback_parser.add_argument("version", help="Version to rollback to")
    
    # latest command
    subparsers.add_parser("latest", help="Return to the latest version")
    
    # init command
    subparsers.add_parser("init", help="Initialize a new project with Git")
    
    # github command
    subparsers.add_parser("github", help="Configure GitHub remote repository")
    
    args = parser.parse_args()
    
    if not args.command or args.command == "save":
        save_version(
            message=args.message if hasattr(args, 'message') else None,
            push=args.push if hasattr(args, 'push') else False,
            auto_increment=not (hasattr(args, 'manual') and args.manual),
            force=args.force if hasattr(args, 'force') else False
        )
    
    elif args.command == "list":
        list_versions()
    
    elif args.command == "rollback":
        rollback(args.version)
    
    elif args.command == "latest":
        return_to_latest()
    
    elif args.command == "init":
        initialize()
    
    elif args.command == "github":
        setup_github_remote()

if __name__ == "__main__":
    main() 