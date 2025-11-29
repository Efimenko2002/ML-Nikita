"""
Код для добавления в notebook - ПОЛНАЯ проверка датасета
Вставьте это в новую ячейку ПЕРЕД обучением модели
"""

# === КОД ДЛЯ NOTEBOOK ===

from PIL import Image
from collections import Counter, defaultdict
import hashlib

print("=" * 80)
print("🔍 ПОЛНАЯ ПРОВЕРКА КАЧЕСТВА ДАТАСЕТА")
print("=" * 80)
print(f"📁 Путь: {config.DATASET_PATH}\n")

# Переменные для сбора статистики
image_sizes = []
corrupted_images = []
image_hashes = defaultdict(list)
class_distribution = {}

print("📂 Анализ структуры и изображений...")

# Проходим по всем классам
for class_name in sorted(os.listdir(config.DATASET_PATH)):
    class_path = os.path.join(config.DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    image_files = [f for f in os.listdir(class_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    class_distribution[class_name] = len(image_files)

    # Проверяем каждое изображение
    for img_name in image_files:
        img_path = os.path.join(class_path, img_name)

        try:
            # Открываем и проверяем изображение
            img = Image.open(img_path)
            img.verify()

            # Переоткрываем для получения данных
            img = Image.open(img_path)
            width, height = img.size
            image_sizes.append((width, height))

            # Вычисляем хеш для поиска дубликатов
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                image_hashes[file_hash].append(img_path)

        except Exception as e:
            corrupted_images.append((img_path, str(e)))

print(f"✅ Проверено изображений: {len(image_sizes)}\n")

# === РЕЗУЛЬТАТЫ ===

print("📊 РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ:")
for cls, count in class_distribution.items():
    print(f"  {cls}: {count} изображений")

counts = list(class_distribution.values())
if len(set(counts)) > 1:
    print(f"\n⚠️  Классы несбалансированы! Мин: {min(counts)}, Макс: {max(counts)}")
else:
    print(f"\n✅ Классы сбалансированы: по {counts[0]} изображений")

print("\n" + "=" * 80)
print("📐 РАЗМЕРЫ ИЗОБРАЖЕНИЙ:")
print("=" * 80)

size_counter = Counter(image_sizes)
for size, count in size_counter.most_common(20):
    is_square = "✅" if size[0] == size[1] else "❌"
    aspect_ratio = size[0] / size[1] if size[1] > 0 else 0
    print(f"  {is_square} {size[0]:4d}×{size[1]:4d} (ratio: {aspect_ratio:.3f}): {count:4d} изображений")

# Статистика квадратности
square_images = sum(1 for w, h in image_sizes if w == h)
total_images = len(image_sizes)

print(f"\n📊 Квадратных: {square_images}/{total_images} ({100*square_images/total_images:.1f}%)")

if square_images < total_images:
    print("⚠️  ПРОБЛЕМА: Есть неквадратные изображения!")
    print("   → Текущий Resize((224, 224)) ИСКАЖАЕТ их!")

# Диапазон размеров
if image_sizes:
    widths = [w for w, h in image_sizes]
    heights = [h for w, h in image_sizes]

    print(f"\n📏 Диапазон размеров:")
    print(f"  Ширина:  {min(widths):4d} - {max(widths):4d} px (средняя: {sum(widths)//len(widths):4d})")
    print(f"  Высота:  {min(heights):4d} - {max(heights):4d} px (средняя: {sum(heights)//len(heights):4d})")

print("\n" + "=" * 80)
print("🔍 ПРОВЕРКА НА ДУБЛИКАТЫ:")
print("=" * 80)

duplicates = {h: paths for h, paths in image_hashes.items() if len(paths) > 1}

if duplicates:
    print(f"⚠️  Найдено дубликатов: {len(duplicates)} групп")
    print(f"   Всего повторяющихся файлов: {sum(len(paths) for paths in duplicates.values())}")

    for i, (file_hash, paths) in enumerate(list(duplicates.items())[:3]):
        print(f"\n  Группа {i+1} ({len(paths)} одинаковых файлов):")
        for path in paths[:3]:
            print(f"    - {path}")
else:
    print("✅ Дубликатов не найдено")

if corrupted_images:
    print(f"\n❌ БИТЫХ ФАЙЛОВ: {len(corrupted_images)}")
    for img_path, error in corrupted_images[:5]:
        print(f"   {img_path}")
        print(f"      Ошибка: {error}")
else:
    print("\n✅ Битых файлов не найдено")

print("\n" + "=" * 80)
print("🚨 ИТОГОВЫЕ РЕКОМЕНДАЦИИ:")
print("=" * 80)

issues = []

if square_images < total_images:
    issues.append("❌ КРИТИЧНО: Неквадратные изображения → меняйте трансформации!")

if len(size_counter) > 10:
    issues.append(f"⚠️  Слишком много разных размеров ({len(size_counter)}) → нестабильное обучение")

if duplicates:
    issues.append(f"⚠️  Есть дубликаты ({len(duplicates)}) → могут влиять на метрики")

if corrupted_images:
    issues.append(f"❌ КРИТИЧНО: Битые файлы ({len(corrupted_images)}) → удалите их!")

if issues:
    for issue in issues:
        print(f"  {issue}")
else:
    print("  ✅ Серьезных проблем не обнаружено!")

print("=" * 80)
