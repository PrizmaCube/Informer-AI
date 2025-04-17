#!/usr/bin/env python
import os
import subprocess
import argparse
import re
import requests
import zipfile
import shutil

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
        return None

def get_github_info():
    """Get GitHub repository information"""
    try:
        remote_url = run_command("git remote get-url origin")
        if not remote_url:
            print("No GitHub remote found.")
            return None, None
        
        # Extract username and repo name from URL
        match = re.search(r'github\.com[:/]([^/]+)/([^/.]+)', remote_url)
        if match:
            username = match.group(1)
            repo = match.group(2)
            return username, repo
        
        print(f"Could not parse GitHub URL: {remote_url}")
        return None, None
        
    except Exception as e:
        print(f"Error getting GitHub info: {e}")
        return None, None

def get_remote_versions(username, repo):
    """Get versions available on GitHub"""
    try:
        url = f"https://api.github.com/repos/{username}/{repo}/tags"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error accessing GitHub API: {response.status_code}")
            print(f"Message: {response.text}")
            return []
        
        tags = response.json()
        return [tag['name'] for tag in tags]
        
    except Exception as e:
        print(f"Error fetching versions from GitHub: {e}")
        return []

def download_version(username, repo, version, destination=None):
    """Download a specific version from GitHub"""
    try:
        # If destination not specified, use current directory
        if not destination:
            destination = os.path.join(os.getcwd(), f"{repo}-{version}")
        else:
            destination = os.path.abspath(destination)
        
        # Create destination directory if it doesn't exist
        if not os.path.exists(destination):
            os.makedirs(destination)
        
        print(f"Downloading version {version} from GitHub...")
        
        # Download the version as zip archive
        zip_url = f"https://github.com/{username}/{repo}/archive/refs/tags/{version}.zip"
        download_path = os.path.join(os.getcwd(), f"{repo}-{version}.zip")
        
        response = requests.get(zip_url, stream=True)
        if response.status_code != 200:
            print(f"Error downloading version: {response.status_code}")
            print(f"Message: {response.text}")
            return False
        
        # Save the zip file
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded to {download_path}")
        
        # Extract the zip file
        print(f"Extracting files to {destination}...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(destination))
        
        # Rename the extracted directory
        extracted_dir = os.path.join(os.path.dirname(destination), f"{repo}-{version}")
        if os.path.exists(extracted_dir) and extracted_dir != destination:
            if os.path.exists(destination):
                shutil.rmtree(destination)
            shutil.move(extracted_dir, destination)
        
        # Remove the zip file
        os.remove(download_path)
        
        print(f"Version {version} successfully downloaded to {destination}")
        return True
        
    except Exception as e:
        print(f"Error downloading version: {e}")
        return False

def fetch_version(version=None, destination=None, current_repo=True):
    """Fetch a specific version from GitHub"""
    if current_repo:
        # Try to get info from current git repo
        username, repo = get_github_info()
        if not username or not repo:
            print("Could not determine GitHub repository information.")
            return False
    else:
        # Ask user for repository information
        print("\nEnter GitHub repository information:")
        username = input("Username: ")
        repo = input("Repository name: ")
        
        if not username or not repo:
            print("Invalid GitHub repository information.")
            return False
    
    # Get available versions
    versions = get_remote_versions(username, repo)
    
    if not versions:
        print(f"No versions found for {username}/{repo}")
        return False
    
    # Print available versions if no version specified
    if not version:
        print("\nAvailable versions:")
        for i, ver in enumerate(versions):
            print(f"{i+1}. {ver}")
        
        selection = input("\nSelect version number or enter version name: ")
        try:
            if selection.isdigit() and 1 <= int(selection) <= len(versions):
                version = versions[int(selection) - 1]
            else:
                version = selection
        except Exception:
            version = selection
    
    # Check if version exists
    if version not in versions:
        print(f"Version {version} not found. Available versions: {', '.join(versions)}")
        return False
    
    # Ask for destination if not specified
    if not destination:
        default_dest = os.path.join(os.getcwd(), f"{repo}-{version}")
        user_dest = input(f"Enter destination directory [default: {default_dest}]: ")
        if user_dest:
            destination = user_dest
        else:
            destination = default_dest
    
    # Download the version
    return download_version(username, repo, version, destination)

def update_local_repo(version=None):
    """Update local repository to a specific version"""
    # Check if we're in a git repository
    if not os.path.isdir(".git"):
        print("Not in a git repository. Please run this command from the repository root.")
        return False
    
    # Fetch all tags from remote
    print("Fetching latest data from GitHub...")
    run_command("git fetch --all --tags")
    
    # Get available tags
    tags = run_command("git tag -l").split('\n')
    if not tags or not tags[0]:
        print("No versions (tags) found in the repository.")
        return False
    
    # Sort tags by version
    tags.sort(key=lambda x: [int(n) for n in x.replace('v', '').split('.')])
    tags.reverse()  # Most recent first
    
    # Print available versions if no version specified
    if not version:
        print("\nAvailable versions:")
        for i, tag in enumerate(tags):
            # Get tag date and message
            info = run_command(f'git show {tag} --pretty=format:"%ci | %s" -s')
            date, message = info.split(" | ", 1) if " | " in info else (info, "")
            print(f"{i+1}. {tag} ({date}) - {message}")
        
        selection = input("\nSelect version number or enter version name: ")
        try:
            if selection.isdigit() and 1 <= int(selection) <= len(tags):
                version = tags[int(selection) - 1]
            else:
                version = selection
        except Exception:
            version = selection
    
    # Check if version exists
    if version not in tags:
        print(f"Version {version} not found locally. Available versions: {', '.join(tags)}")
        return False
    
    # Check for local changes
    changes = run_command("git status --porcelain")
    if changes:
        save = input("There are unsaved local changes. Do you want to stash them? (y/n): ")
        if save.lower() == 'y':
            run_command("git stash save 'Automatic stash before version switch'")
            print("Local changes have been stashed. Use 'git stash pop' to restore them later.")
    
    # Switch to the requested version
    print(f"Switching to version {version}...")
    result = run_command(f"git checkout {version}")
    
    if result is not None:
        print(f"Successfully switched to version {version}")
        print("\nNOTE: You are now in 'detached HEAD' state.")
        print("To return to the latest version, use: python git_download.py --latest")
        return True
    else:
        print(f"Failed to switch to version {version}")
        return False

def return_to_latest():
    """Return to the latest version (master/main branch)"""
    # Try to determine the default branch
    for branch in ["master", "main"]:
        if run_command(f"git show-ref --verify --quiet refs/heads/{branch}"):
            run_command(f"git checkout {branch}")
            print(f"Returned to the latest version (branch: {branch})")
            return True
    
    # If default branches don't exist, try another approach
    branches = run_command("git branch").split('\n')
    if branches:
        # Find the current branch (has a * prefix)
        current_branch = next((b[2:] for b in branches if b.startswith('* ')), None)
        if current_branch and current_branch != "HEAD":
            run_command(f"git checkout {current_branch}")
            print(f"Returned to the latest version (branch: {current_branch})")
            return True
    
    print("Could not determine the default branch. Please manually checkout your main branch.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Download specific versions from GitHub or update local repository")
    parser.add_argument("version", nargs="?", help="Version to download/switch to (optional)")
    parser.add_argument("--download", action="store_true", help="Download the version instead of switching local repo")
    parser.add_argument("--dest", help="Destination directory for downloads")
    parser.add_argument("--repo", action="store_true", help="Specify a different GitHub repository")
    parser.add_argument("--latest", action="store_true", help="Return to the latest version (master/main branch)")
    
    args = parser.parse_args()
    
    if args.latest:
        return_to_latest()
    elif args.download or args.repo:
        fetch_version(args.version, args.dest, not args.repo)
    else:
        update_local_repo(args.version)

if __name__ == "__main__":
    main() 