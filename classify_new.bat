@chcp 65001 >nul
@echo off
echo ========================================
echo Классификация новых товаров
echo ========================================
echo.

call venv\Scripts\activate
python main.py --mode predict

echo.
echo Готово! Результаты в папке data\processed\
pause