@echo off
echo ========================================
echo Улучшение модели на основе проверок
echo ========================================
echo.

call venv\Scripts\activate
python main.py --mode active

echo.
echo Модель обновлена с учётом ваших правок!
pause