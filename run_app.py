#!/usr/bin/env python3
"""
Face Vital Monitor - Application Launcher
Run this script to launch the comprehensive health monitoring application.
"""

import subprocess
import sys
import os


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "streamlit",
        "opencv-python",
        "mediapipe",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "plotly",
        "reportlab",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    print("✅ All dependencies are installed!")
    return True


def main():
    """Main launcher function"""
    print("🫀 Face Vital Monitor - Application Launcher")
    print("=" * 50)

    # # Check dependencies
    # if not check_dependencies():
    #     sys.exit(1)

    print("\nAvailable applications:")
    print("1. 🏥 Comprehensive Health Monitor (streamlit_comprehensive_app.py)")
    print("2. 📊 Advanced Health Scanner (streamlit_app.py)")
    print("3. 🎯 Simple AI Health Scanner (streamlit_facemesh.py)")
    print("4. 🖥️  Original Tkinter Version (facemesh.py)")

    app_file = "streamlit_app.py"

    # Launch selected Streamlit app
    print(f"🚀 Launching {app_file}...")
    print("📱 The application will open in your default web browser.")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n⚠️  Important:")
    print("   - Allow camera access when prompted")
    print("   - Ensure good lighting on your face")
    print("   - Stay still during monitoring sessions")
    print("\n🛑 Press Ctrl+C in this terminal to stop the application")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                app_file,
                "--server.address",
                "localhost",
                "--server.port",
                "8501",
                "--browser.gatherUsageStats",
                "false",
            ]
        )
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
    except Exception as e:
        print(f"❌ Error launching application: {e}")


if __name__ == "__main__":
    main()
