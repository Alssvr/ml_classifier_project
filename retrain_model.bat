@chcp 65001 >nul
@echo off
echo ========================================
echo Переобучение модели на новых данных
echo ========================================
echo.

call venv\Scripts\activate
python main.py --mode train

echo.
echo Модель переобучена!
pause