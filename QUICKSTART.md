# 🚀 Быстрый старт

Краткая инструкция по запуску проекта Vision Transformer для классификации изображений.

---

## ⚡ За 5 минут

### 1. Подготовка датасета

Организуйте датасет на Google Drive:

```
Google Drive/
└── radar_dataset/
    ├── class_0/
    │   ├── img001.jpg
    │   └── img002.jpg
    ├── class_1/
    ├── class_2/
    ├── class_3/
    ├── class_4/
    └── class_5/
```

### 2. Откройте notebook в Google Colab

1. Перейдите на [Google Colab](https://colab.research.google.com/)
2. `File → Upload notebook`
3. Загрузите `vision_transformer_classification.ipynb`
4. Включите GPU: `Runtime → Change runtime type → GPU → Save`

### 3. Укажите путь к датасету

Найдите ячейку с настройкой пути и измените:

```python
config.DATASET_PATH = '/content/drive/MyDrive/radar_dataset'  # ← ВАШ ПУТЬ
```

### 4. Запустите все ячейки

```
Runtime → Run all
```

или нажмите `Ctrl+F9` (Windows) / `Cmd+F9` (Mac)

### 5. Подождите завершения обучения

- ⏱️ ~1-1.5 часа на GPU T4
- 📊 Метрики выводятся после каждой эпохи
- 💾 Лучшая модель автоматически сохраняется

### 6. Получите результаты

После обучения автоматически создаются:
- ✅ `best_model.pth` — лучшая модель
- 📊 `training_history.png` — графики обучения
- 📊 `confusion_matrix.png` — матрица ошибок
- 🔍 `attention_maps.png` — визуализация attention

---

## 📝 Использование обученной модели

### Предсказание в notebook

```python
# Загрузка модели
checkpoint = torch.load('vit_radar_classifier_final.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Предсказание
predicted_class, confidence, all_probs, image = predict_image(
    model, 'path/to/image.jpg', val_test_transforms, device, class_names
)

print(f"Класс: {predicted_class}")
print(f"Уверенность: {confidence:.2f}%")
```

### Локальный inference (на своем компьютере)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск предсказания
python inference_example.py --image path/to/image.jpg --model vit_radar_classifier_final.pth
```

---

## ⚙️ Основные параметры

Измените в классе `Config`:

```python
class Config:
    NUM_CLASSES = 6          # Количество классов
    IMAGE_SIZE = 224         # Размер изображения
    BATCH_SIZE = 16          # Уменьшите если не хватает памяти GPU
    NUM_EPOCHS = 50          # Максимум эпох
    LEARNING_RATE = 3e-4     # Скорость обучения
    PATIENCE = 10            # Early stopping patience
```

---

## 🐛 Частые проблемы

### ❌ Не хватает памяти GPU

**Решение:** Уменьшите `BATCH_SIZE`:
```python
BATCH_SIZE = 8  # или даже 4
```

### ❌ Датасет не найден

**Решение:** Проверьте путь:
```python
# Должен быть полный путь к папке с классами
config.DATASET_PATH = '/content/drive/MyDrive/radar_dataset'
```

### ❌ CUDA out of memory

**Решение:** Перезапустите runtime и уменьшите batch size:
```
Runtime → Restart runtime
```

---

## 📊 Ожидаемые результаты

Для датасета с 273 изображениями на класс:

- ✅ **Train Accuracy:** 85-95%
- ✅ **Validation Accuracy:** 75-90%
- ✅ **Test Accuracy:** 75-90%
- ⏱️ **Время обучения:** 1-1.5 часа (GPU T4)
- 🔄 **Эпох:** 15-30 (с early stopping)

---

## 💡 Советы для лучших результатов

1. **Больше данных = лучше результат**
   - Соберите больше изображений для каждого класса

2. **Data Augmentation уже включена**
   - Flip, rotation, color jitter применяются автоматически

3. **Fine-tuning работает лучше**
   - Используется предобученная модель на ImageNet

4. **Мониторьте метрики**
   - Если val accuracy не растет — возможно переобучение
   - Early stopping остановит обучение автоматически

5. **Экспериментируйте**
   - Попробуйте разные learning rates
   - Измените количество эпох
   - Используйте другую предобученную модель

---

## 📚 Дальнейшее чтение

- [README.md](README.md) — полная документация с теорией
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929) — оригинальная статья
- [HuggingFace ViT](https://huggingface.co/docs/transformers/model_doc/vit) — документация модели

---

## 🆘 Нужна помощь?

1. Прочитайте [README.md](README.md) — там подробная информация
2. Проверьте раздел FAQ в README.md
3. Убедитесь что структура датасета правильная
4. Проверьте что GPU включен в Colab

---

**Удачи! 🎉**
