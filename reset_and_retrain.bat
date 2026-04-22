@echo off
echo ========================================
echo ПОЛНЫЙ СБРОС И ПЕРЕОБУЧЕНИЕ МОДЕЛИ
echo ========================================
echo.

echo [1/4] Удаляю старую модель...
if exist models\product_classifier*.pkl del models\product_classifier*.pkl
echo Готово.

echo [2/4] Очищаю временные файлы...
if exist data\processed\train_processed.xlsx del data\processed\train_processed.xlsx
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo Готово.

echo [3/4] Активирую окружение...
call venv\Scripts\activate

echo [4/4] Запускаю обучение на НОВЫХ данных...
python main.py --mode train

echo.
echo ========================================
echo ОБУЧЕНИЕ ЗАВЕРШЕНО!
echo ========================================
echo.
echo Проверьте accuracy выше. Если обучение прошло успешно,
echo новая модель сохранена в папке models\
echo.
pause