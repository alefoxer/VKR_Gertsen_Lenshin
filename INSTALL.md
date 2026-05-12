# Установка GRADV

Эта инструкция рассчитана на пользователя, который скачал проект с GitHub или получил ZIP-архив.

## Требования

- Windows 10/11, Linux или macOS.
- Python 3.10-3.12.
- 4 ГБ RAM или больше.
- Интернет для первой установки зависимостей.

На Windows удобнее ставить Python с сайта:

```text
https://www.python.org/downloads/
```

Во время установки отметьте пункт `Add Python to PATH`.

## Установка на Windows

Откройте PowerShell в папке проекта и выполните:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если PowerShell не разрешает активировать окружение:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Установка на Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Запуск

```powershell
python app.py
```

Откройте адрес:

```text
http://127.0.0.1:7860
```

Если приложение выбрало другой порт, используйте адрес из терминала.

## Проверка

```powershell
python scripts\smoke_test.py
```

Если проверка завершилась строкой `[OK] smoke test completed`, установка работает.

## Частые проблемы

### Python не найден

Переустановите Python и включите `Add Python to PATH`, затем откройте новый терминал.

### Не ставится PyTorch

Обновите `pip`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если ошибка сохраняется, используйте Python 3.10 или 3.11.

### Адрес не открывается

Проверьте, что терминал с `python app.py` не закрыт. Если порт занят, приложение напечатает другой адрес.

### Запуск с другого устройства

Используйте инструкцию:

```text
docs/SHARING_GUIDE.md
```
