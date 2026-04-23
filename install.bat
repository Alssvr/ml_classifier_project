@chcp 65001 >nul
@echo off
echo ========================================
echo Установка классификатора товаров
echo ========================================
echo.

echo [1/3] Создаю виртуальное окружение...
python -m venv venv

echo [2/3] Активирую окружение...
call venv\Scripts\activate

echo [3/3] Устанавливаю зависимости...
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements.txt

echo.
echo ========================================
echo Установка завершена!
echo ========================================
echo.
echo Теперь можно:
echo - classify_new.bat - классифицировать новые товары
echo - improve_model.bat - улучшить модель после проверки
echo - retrain_model.bat - переобучить на новых данных
echo.
pause