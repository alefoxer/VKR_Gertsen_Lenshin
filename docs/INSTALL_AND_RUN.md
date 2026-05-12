# Установка и запуск

## 1. Подготовка

Установите Python 3.10-3.12. На Windows при установке включите `Add Python to PATH`.

Проверьте версию:

```powershell
python --version
```

## 2. Установка зависимостей

Откройте терминал в папке проекта:

```powershell
cd "C:\path\to\Speech3_final"
```

Создайте окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Для Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Проверка

```powershell
python scripts\smoke_test.py
```

Ожидаемый результат:

```text
[OK] smoke test completed
```

## 4. Запуск приложения

```powershell
python app.py
```

Откройте:

```text
http://127.0.0.1:7860
```

Если порт занят, приложение напечатает другой локальный адрес.

## 5. Baseline-файлы

Обычно baseline уже лежит в папке `models/`:

```text
models\gradv_ru_kws_baseline.pt
models\gradv_ru_kws_vocab.txt
models\gradv_ru_kws_training_summary.json
```

Если файлов нет:

```powershell
python experiments\train_gradv_ru_kws_baseline.py
```

## 6. Быстрая работа в интерфейсе

1. Нажмите `Применить пресет ВКР baseline quality`.
2. Нажмите `Найти максимизирующий сигнал`.
3. Дождитесь карточек результата.
4. Откройте графики, таблицы сегментов и файлы экспорта.

## 7. Где сохраняются результаты

```text
outputs/runs/
outputs/studies/
outputs/model_comparisons/
outputs/reports/
```

Папка `outputs/` не входит в GitHub-экспорт, потому что содержит результаты запусков.
