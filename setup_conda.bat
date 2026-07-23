@echo off
REM ============================================================
REM  DCVC-RT: Auto Setup Script (Miniconda + PyTorch CUDA + Test)
REM  Chạy script này sau khi cài Miniconda xong
REM ============================================================

echo ============================================================
echo  DCVC-RT Auto Setup
echo ============================================================

REM Tìm đường dẫn Miniconda
SET CONDA_PATH=%USERPROFILE%\miniconda3
IF NOT EXIST "%CONDA_PATH%\Scripts\conda.exe" (
    SET CONDA_PATH=%USERPROFILE%\Miniconda3
)
IF NOT EXIST "%CONDA_PATH%\Scripts\conda.exe" (
    echo ERROR: Miniconda not found at %CONDA_PATH%
    echo Please install Miniconda first or set CONDA_PATH manually
    pause
    exit /b 1
)

echo Found Conda at: %CONDA_PATH%
call "%CONDA_PATH%\Scripts\activate.bat" "%CONDA_PATH%"

REM Tạo môi trường Python 3.12
echo.
echo [1/5] Creating conda environment 'dcvc' with Python 3.12...
call conda create -n dcvc python=3.12 -y

REM Kích hoạt môi trường
echo.
echo [2/5] Activating environment...
call conda activate dcvc

REM Cài PyTorch CUDA
echo.
echo [3/5] Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Cài requirements
echo.
echo [4/5] Installing other requirements...
pip install -r requirements.txt

REM Kiểm tra
echo.
echo [5/5] Verifying installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo ============================================================
echo  Setup complete!
echo  Next steps:
echo    1. Download checkpoints to checkpoints/ folder
echo    2. Run: python generate_test_video.py
echo    3. Run: python test_video.py ...
echo ============================================================
pause
