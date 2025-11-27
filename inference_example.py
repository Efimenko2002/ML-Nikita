"""
Пример скрипта для inference (предсказания) на обученной модели
Использование: python inference_example.py --image path/to/image.jpg
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
import numpy as np


def load_model(model_path, device='cuda'):
    """
    Загрузка обученной модели

    Args:
        model_path: путь к сохраненной модели
        device: устройство (cuda/cpu)

    Returns:
        model: загруженная модель
        class_names: список названий классов
        config: конфигурация модели
    """
    print(f"📥 Загрузка модели из {model_path}...")

    # Загружаем checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Получаем конфигурацию
    config = checkpoint['config']
    class_names = checkpoint['class_names']

    # Создаем модель
    model = ViTForImageClassification.from_pretrained(
        config['model_name'],
        num_labels=config['num_classes'],
        ignore_mismatched_sizes=True
    )

    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✅ Модель загружена!")
    print(f"📊 Классов: {config['num_classes']}")
    print(f"🏷️  Названия: {class_names}")

    return model, class_names, config


def get_transforms(image_size=224):
    """
    Создание трансформаций для изображения

    Args:
        image_size: размер изображения

    Returns:
        transform: композиция трансформаций
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict(model, image_path, transform, device, class_names):
    """
    Предсказание класса для изображения

    Args:
        model: обученная модель
        image_path: путь к изображению
        transform: трансформации
        device: устройство
        class_names: список названий классов

    Returns:
        predicted_class: предсказанный класс
        confidence: уверенность (%)
        all_probs: вероятности всех классов
        image: оригинальное изображение
    """
    # Загружаем изображение
    image = Image.open(image_path).convert('RGB')

    # Применяем трансформации
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        logits = outputs.logits

        # Применяем softmax для получения вероятностей
        probs = torch.nn.functional.softmax(logits, dim=1)

        # Получаем предсказанный класс и уверенность
        confidence, predicted = torch.max(probs, 1)

    predicted_class = class_names[predicted.item()]
    confidence = confidence.item() * 100
    all_probs = probs[0].cpu().numpy() * 100

    return predicted_class, confidence, all_probs, image


def visualize_prediction(image, predicted_class, confidence, all_probs, class_names, save_path=None):
    """
    Визуализация предсказания

    Args:
        image: изображение (PIL)
        predicted_class: предсказанный класс
        confidence: уверенность
        all_probs: вероятности всех классов
        class_names: названия классов
        save_path: путь для сохранения (опционально)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Изображение ===
    axes[0].imshow(image)
    axes[0].set_title(
        f'Предсказание: {predicted_class}\nУверенность: {confidence:.2f}%',
        fontsize=14,
        fontweight='bold'
    )
    axes[0].axis('off')

    # === Bar plot с вероятностями ===
    y_pos = np.arange(len(class_names))
    bars = axes[1].barh(y_pos, all_probs, color='skyblue')

    # Выделяем предсказанный класс красным
    predicted_idx = class_names.index(predicted_class)
    bars[predicted_idx].set_color('red')

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel('Вероятность (%)', fontsize=12)
    axes[1].set_title('Распределение вероятностей', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 100)

    # Добавляем значения на баре
    for i, (bar, prob) in enumerate(zip(bars, all_probs)):
        axes[1].text(
            prob + 2, i, f'{prob:.1f}%',
            va='center', fontsize=10
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Визуализация сохранена: {save_path}")

    plt.show()


def main():
    """Основная функция"""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Классификация изображения с помощью Vision Transformer'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Путь к изображению для классификации'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='vit_radar_classifier_final.pth',
        help='Путь к обученной модели (по умолчанию: vit_radar_classifier_final.pth)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Устройство для вычислений (cuda/cpu)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Путь для сохранения визуализации (опционально)'
    )

    args = parser.parse_args()

    # Проверка доступности CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA недоступна, используем CPU")
        args.device = 'cpu'

    print(f"🖥️  Устройство: {args.device}")

    # Загрузка модели
    model, class_names, config = load_model(args.model, args.device)

    # Создание трансформаций
    transform = get_transforms(config['image_size'])

    # Предсказание
    print(f"\n🔍 Анализ изображения: {args.image}")
    predicted_class, confidence, all_probs, image = predict(
        model, args.image, transform, args.device, class_names
    )

    # Вывод результатов
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
    print("=" * 60)
    print(f"🎯 Предсказанный класс: {predicted_class}")
    print(f"📈 Уверенность: {confidence:.2f}%")
    print("\n📋 Вероятности всех классов:")
    for cls, prob in zip(class_names, all_probs):
        print(f"  {cls:15s}: {prob:6.2f}%")
    print("=" * 60)

    # Визуализация
    visualize_prediction(
        image, predicted_class, confidence, all_probs, class_names, args.save
    )


if __name__ == '__main__':
    main()
