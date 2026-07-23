@echo off
REM ============================================================
REM  DCVC-RT Setup & Test Script
REM  Run this script to set up environment and test the model
REM ============================================================

echo ============================================================
echo  DCVC-RT - Setup and Test Script
echo ============================================================
echo.

REM Step 1: Check Python
echo [Step 1/6] Checking Python version...
python --version
echo.

REM Step 2: Check/Install PyTorch CUDA
echo [Step 2/6] Checking PyTorch CUDA...
python -c "import torch; print('PyTorch:', torch.__version__); cuda=torch.cuda.is_available(); print('CUDA available:', cuda); exit(0 if cuda else 1)"
if %errorlevel% neq 0 (
    echo.
    echo WARNING: PyTorch CUDA is NOT available!
    echo Please install PyTorch with CUDA support first:
    echo.
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo If your Python version is too new, try:
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    echo.
    pause
    exit /b 1
)
echo.

REM Step 3: Install requirements
echo [Step 3/6] Installing requirements...
pip install -r requirements.txt
echo.

REM Step 4: Generate test video
echo [Step 4/6] Generating synthetic test video...
python generate_test_video.py
echo.

REM Step 5: Check checkpoints
echo [Step 5/6] Checking pretrained checkpoints...
if not exist "checkpoints\cvpr2025_image.pth.tar" (
    echo.
    echo ERROR: Pretrained model checkpoints not found!
    echo Please download them from:
    echo   https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8
    echo.
    echo And place these files in the checkpoints folder:
    echo   checkpoints\cvpr2025_image.pth.tar
    echo   checkpoints\cvpr2025_video.pth.tar
    echo.
    pause
    exit /b 1
)
echo Checkpoints found!
echo.

REM Step 6: Run test
echo [Step 6/6] Running DCVC-RT test on small video...
echo ============================================================
python test_video.py ^
    --model_path_i ./checkpoints/cvpr2025_image.pth.tar ^
    --model_path_p ./checkpoints/cvpr2025_video.pth.tar ^
    --rate_num 2 ^
    --test_config ./dataset_config_test_small.json ^
    --cuda 1 ^
    -w 1 ^
    --write_stream 1 ^
    --force_zero_thres 0.12 ^
    --output_path output/test_result.json ^
    --force_intra_period -1 ^
    --reset_interval 64 ^
    --force_frame_num 10 ^
    --check_existing 0 ^
    --verbose 2

echo.
echo ============================================================
echo Test complete! Check output/test_result.json for results.
echo ============================================================
pause
