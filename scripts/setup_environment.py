#!/usr/bin/env python3
"""
Setup script for chest X-ray resolution impact project.

This script automates the setup of the virtual environment and installation
of all required dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {command}")
    print(f"{'='*60}")
    result = subprocess.run(
        command,
        shell=True,
        check=check,
        capture_output=False
    )
    return result


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")


def create_venv(venv_name="venv"):
    """Create a virtual environment."""
    if os.path.exists(venv_name):
        print(f"\nVirtual environment '{venv_name}' already exists.")
        response = input("Do you want to recreate it? (y/n): ").strip().lower()
        if response == 'y':
            print(f"Removing existing virtual environment...")
            if platform.system() == "Windows":
                run_command(f"rmdir /s /q {venv_name}", check=False)
            else:
                run_command(f"rm -rf {venv_name}", check=False)
        else:
            print("Using existing virtual environment.")
            return
    
    print(f"\nCreating virtual environment: {venv_name}")
    run_command(f"python -m venv {venv_name}")
    print(f"✓ Virtual environment created successfully!")


def install_requirements(venv_name="venv"):
    """Install requirements in the virtual environment."""
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_name, "Scripts", "pip")
        python_path = os.path.join(venv_name, "Scripts", "python")
    else:
        pip_path = os.path.join(venv_name, "bin", "pip")
        python_path = os.path.join(venv_name, "bin", "python")
    
    print(f"\nUpgrading pip...")
    run_command(f'"{python_path}" -m pip install --upgrade pip')
    
    print(f"\nInstalling requirements...")
    if not os.path.exists("requirements.txt"):
        print("ERROR: requirements.txt not found!")
        sys.exit(1)
    
    # Try standard installation first
    try:
        run_command(f'"{pip_path}" install -r requirements.txt')
        print("✓ All requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nWarning: Some packages may have failed to install.")
        print("This is common with packages like pycocotools on Windows.")
        print("Please check the error messages above and install manually if needed.")
        return False
    
    return True


def verify_installation(venv_name="venv"):
    """Verify that key packages are installed."""
    if platform.system() == "Windows":
        python_path = os.path.join(venv_name, "Scripts", "python")
    else:
        python_path = os.path.join(venv_name, "bin", "python")
    
    print(f"\n{'='*60}")
    print("Verifying installation...")
    print(f"{'='*60}")
    
    # Test imports
    test_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("ultralytics", "Ultralytics"),
        ("wandb", "WandB"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
    ]
    
    success = True
    for module, name in test_imports:
        try:
            result = subprocess.run(
                [python_path, "-c", f"import {module}"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"✓ {name} installed")
            else:
                print(f"✗ {name} NOT installed")
                success = False
        except Exception:
            print(f"✗ {name} NOT installed")
            success = False
    
    return success


def print_activation_instructions(venv_name="venv"):
    """Print instructions for activating the virtual environment."""
    print(f"\n{'='*60}")
    print("SETUP COMPLETE!")
    print(f"{'='*60}")
    print("\nTo activate your virtual environment:")
    
    if platform.system() == "Windows":
        print(f"  {venv_name}\\Scripts\\activate")
    else:
        print(f"  source {venv_name}/bin/activate")
    
    print("\nTo deactivate when you're done:")
    print("  deactivate")
    print(f"\n{'='*60}\n")


def main():
    """Main setup function."""
    print("="*60)
    print("Chest X-ray Resolution Impact - Environment Setup")
    print("="*60)
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    venv_name = "venv"
    create_venv(venv_name)
    
    # Install requirements
    install_requirements(venv_name)
    
    # Verify installation
    verify_installation(venv_name)
    
    # Print final instructions
    print_activation_instructions(venv_name)


if __name__ == "__main__":
    main()
