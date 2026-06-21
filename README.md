[![DOI](https://zenodo.org/badge/1236887283.svg)](https://doi.org/10.5281/zenodo.20768313)
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
## Reproducible manuscript experiment



The reproducible experiment used for the manuscript can be run with:



```bash

python experiments/run_final_vkr_experiment.py

```



The script generates a complete experimental output package, including:



- single-run results;

- study-mode outputs;

- model-comparison summaries;

- WAV, CSV, JSON, and Markdown artifacts;

- `final_experiment_summary.json`;

- `final_experiment_report.md`.



The generated outputs are saved under:



```text

outputs/final_vkr_experiment/<experiment_id>/

```

## Manuscript extended experiment

The extended manuscript experiment evaluates target-class audio pattern discovery across all available vocabulary targets, repeated seeds and the `maximize_from_noise` / `maximize_from_silence` input modes. It exports machine-readable CSV/JSON summaries, segment-ranking ablation, weight-sensitivity analysis, optional model-comparison outputs and manuscript-ready figures.

Quick verification command:

```bash
python experiments/run_manuscript_extended_experiment.py --targets да нет --steps 8 --repeats 1 --quick
```

Full manuscript command:

```bash
python experiments/run_manuscript_extended_experiment.py --steps 50 --repeats 5 --input-modes maximize_from_noise maximize_from_silence --seed-start 8000
```

Generated outputs are saved under `outputs/manuscript_extended_experiment/`. See [docs/MANUSCRIPT_EXTENDED_EXPERIMENT.md](docs/MANUSCRIPT_EXTENDED_EXPERIMENT.md) for the complete reviewer-oriented protocol and file descriptions.



## Citation



If you use GRADV in academic work, please cite the archived software release. The DOI will be added after the Zenodo release.



```text

Gertsen, A. M., & Lenshin, A. M. (2026). GRADV: A Reproducible Pipeline for Interpretable Audio Pattern Discovery in Keyword-Spotting Models (v1.0.1). Zenodo. DOI: 10.5281/zenodo.20768314

```



## License



This project is distributed under the MIT License. See the `LICENSE` file for details.



## Manuscript scope



The current release is intended for reproducible research on interpretable audio pattern discovery in keyword-spotting models. The included `gradv_ru_kws_baseline` model is a compact demonstration model. The results should be interpreted as the behavior of a specific model and selected experimental parameters, not as a universal description of Russian speech or all speech-recognition systems.
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
