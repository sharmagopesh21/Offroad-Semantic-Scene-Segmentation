@echo off
echo =====================================
echo Creating virtual environment...
echo =====================================

python -m venv venv

if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate

if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing required packages...
python -m pip install torch torchvision numpy matplotlib pillow opencv-python

echo.
echo =====================================
echo Setup completed successfully!
echo =====================================
pause
