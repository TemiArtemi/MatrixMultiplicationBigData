import sys
import subprocess
import os
from pathlib import Path

def main():
    # 1. Define absolute paths robustly
    # __file__ points to .../assignment1/scripts/validate_all.py
    # .parent is .../assignment1/scripts
    # .parent.parent is .../assignment1
    ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
    
    PYTHON_DIR = ASSIGNMENT_DIR / "src" / "python"
    JAVA_DIR = ASSIGNMENT_DIR / "src" / "java"
    C_DIR = ASSIGNMENT_DIR / "src" / "c"

    print(f"Base directory: {ASSIGNMENT_DIR}")
    print(f"Python dir: {PYTHON_DIR}")
    print(f"Java dir: {JAVA_DIR}")
    print(f"C dir: {C_DIR}\n")
    
    print("ASSIGNMENT 1 - CORRECTNESS VALIDATION")
    print("Running validation for all three implementations...\n")

    # 2. Configure executable names based on the OS
    c_exe_name = "matrix_mult.exe" if os.name == 'nt' else "matrix_mult"
    c_exe_path = C_DIR / c_exe_name

    # 3. Define the sequence of tasks to execute
    steps = [
        {
            "name": "VALIDATING: Python implementation",
            # Using sys.executable ensures it uses the same Python version 
            # launching this script (prevents environment issues)
            "cmd": [sys.executable, "matrix_mult.py", "--validate-only"],
            "cwd": PYTHON_DIR
        },
        {
            "name": "COMPILING: Java implementation",
            "cmd": ["javac", "MatrixMult.java"],
            "cwd": JAVA_DIR
        },
        {
            "name": "VALIDATING: Java implementation",
            "cmd": ["java", "MatrixMult", "--validate-only"],
            "cwd": JAVA_DIR
        },
        {
            "name": "COMPILING: C implementation",
            # Using -O3 as it will also be required for the benchmarking phase
            "cmd": ["gcc", "-Wall", "-O3", "matrix_mult.c", "-o", c_exe_name],
            "cwd": C_DIR
        },
        {
            "name": "VALIDATING: C implementation",
            # Passing the absolute path of the executable prevents issues with "./" or ".\" on Windows
            "cmd": [str(c_exe_path), "--validate-only"],
            "cwd": C_DIR
        }
    ]

    all_passed = True

    # 4. Execute each step sequentially
    for step in steps:
        print("=" * 60)
        print(step["name"])
        print(f"Command: {' '.join(step['cmd'])}")
        print(f"Directory: {step['cwd']}")
        print("=" * 60)
        
        try:
            # text=True ensures stdout and stderr are treated as strings
            # check=True raises an exception if the process exits with an error
            subprocess.run(
                step["cmd"], 
                cwd=step["cwd"], 
                text=True, 
                check=True
            )
            print(f"✅ {step['name'].split(':')[0]} successful.\n")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Command failed with exit code {e.returncode}.\n")
            all_passed = False
            break  # Stops execution if compilation or validation fails
            
        except FileNotFoundError as e:
            print(f"❌ ERROR: Command '{step['cmd'][0]}' not found.")
            print("   Make sure the tool is installed and added to the system PATH.\n")
            all_passed = False
            break
            
        except NotADirectoryError as e:
            print(f"❌ ERROR: Invalid directory: {e}\n")
            all_passed = False
            break

    # 5. Final summary
    if all_passed:
        print("\n🎉 Success! All compilations and validations ran correctly.")
    else:
        print("\n⚠️ There were errors during the process. Please fix the issues indicated above.")
        sys.exit(1)

if __name__ == "__main__":
    main()