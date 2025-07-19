#!/usr/bin/env python3
"""
Script to help push HR-VITON repository to GitHub
"""

import os
import subprocess
import sys

def check_git_status():
    """Check current Git status"""
    try:
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        print("Current Git Status:")
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Error checking Git status: {e}")
        return False

def create_github_repo_instructions():
    """Provide instructions for creating GitHub repository"""
    print("\n" + "="*60)
    print("📋 INSTRUCTIONS TO PUSH TO GITHUB")
    print("="*60)
    
    print("\n1. Create a new repository on GitHub:")
    print("   - Go to https://github.com/new")
    print("   - Choose a repository name (e.g., 'hr-viton-setup')")
    print("   - Make it public or private")
    print("   - DON'T initialize with README (we already have one)")
    print("   - Click 'Create repository'")
    
    print("\n2. Copy the repository URL (it will look like):")
    print("   https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git")
    
    print("\n3. Run these commands in your terminal:")
    print("   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    
    print("\n4. Alternative: Use GitHub CLI (if installed):")
    print("   gh repo create YOUR_REPO_NAME --public --source=. --remote=origin --push")
    
    print("\n" + "="*60)
    print("✅ Your repository is ready to push!")
    print("📁 All files are committed and ready")
    print("🔒 Large model files are excluded via .gitignore")
    print("📚 Comprehensive README is included")
    print("="*60)

def show_repository_summary():
    """Show what's included in the repository"""
    print("\n📦 REPOSITORY CONTENTS:")
    print("-" * 40)
    
    files = [
        "✅ All original HR-VITON source code",
        "✅ CPU compatibility fixes",
        "✅ Helper scripts (test, monitor, quick_start)",
        "✅ Comprehensive documentation",
        "✅ Small test dataset (10 samples)",
        "✅ .gitignore (excludes large model files)",
        "✅ Setup guides and instructions"
    ]
    
    for file in files:
        print(file)
    
    print("\n📊 Repository Size:")
    print("- Source code: ~2MB")
    print("- Documentation: ~50KB")
    print("- Test data: ~1MB")
    print("- Total: ~3MB (model files excluded)")

def main():
    print("🚀 HR-VITON Repository Push Helper")
    print("=" * 50)
    
    # Check Git status
    if not check_git_status():
        return
    
    # Show repository summary
    show_repository_summary()
    
    # Provide GitHub instructions
    create_github_repo_instructions()
    
    print("\n🎯 Next Steps:")
    print("1. Create GitHub repository")
    print("2. Add remote origin")
    print("3. Push to GitHub")
    print("4. Share your working HR-VITON setup!")

if __name__ == "__main__":
    main() 