# GRADV

GRADV - локальное приложение для анализа аудиосигналов в задаче распознавания коротких русских команд. Программа формирует входной аудиообраз выбранного класса, показывает изменение score, вероятности классов, графики сигнала и таблицы значимых временных участков.

## Быстрый запуск

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Откройте в браузере:

```text
http://127.0.0.1:7860
```

Если порт `7860` занят, приложение выберет ближайший свободный порт и напечатает новый адрес в терминале.

## Что есть в приложении

- одиночный запуск для выбранного слова;
- работа с загруженным WAV/MP3 или стартом из тишины/шума;
- методы `gradient_ascent`, `mask_attack`, `template_projection` и другие варианты оптимизации;
- графики waveform, spectrogram, saliency, probability;
- таблицы временных сегментов;
- экспорт WAV, CSV и JSON;
- study-режим для нескольких повторов;
- сравнение моделей и условий запуска.

## Рекомендуемый сценарий

1. Запустите `python app.py`.
2. Откройте `http://127.0.0.1:7860`.
3. Нажмите `Применить пресет ВКР baseline quality`.
4. Проверьте, что выбраны:
   - модель: `gradv_ru_kws_baseline`;
   - целевое слово: `да`;
   - сценарий: `Сгенерировать из шума`;
   - метод: `Gradient ascent`;
   - objective: `logit`.
5. Нажмите `Найти максимизирующий сигнал`.
6. Посмотрите карточки результата, графики и таблицу сегментов.

## Проверка установки

```powershell
python scripts\smoke_test.py
```

Успешный результат:

```text
[OK] smoke test completed
```

Для более полной проверки:

```powershell
python scripts\verify_core.py
```

## Baseline-модель

В проект входит компактная baseline-модель:

```text
models\gradv_ru_kws_baseline.pt
models\gradv_ru_kws_vocab.txt
models\gradv_ru_kws_training_summary.json
```

Если файлов нет, модель можно пересоздать:

```powershell
python experiments\train_gradv_ru_kws_baseline.py
```

## Экспорт результатов

Основные результаты сохраняются в `outputs/`:

```text
outputs/runs/<run_id>/
outputs/studies/<study_id>/
outputs/model_comparisons/<comparison_id>/
outputs/reports/<report_id>/
```

Для подготовки чистой копии проекта под загрузку в GitHub:

```powershell
python scripts\prepare_github_export.py
```

Скрипт создаст папку и ZIP в:

```text
outputs/github_export/
```

## Документация

- [Установка и запуск](docs/INSTALL_AND_RUN.md)
- [Доступ с другого устройства](docs/SHARING_GUIDE.md)
- [Сценарий демонстрации](docs/VKR_DEMO_GUIDE.md)
- [Краткая инструкция установки](INSTALL.md)

## Ограничения

`gradv_ru_kws_baseline` - компактная демонстрационная KWS-модель. Результаты следует читать как поведение конкретной модели и выбранных параметров, а не как универсальное описание русской речи.
