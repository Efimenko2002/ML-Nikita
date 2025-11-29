"""
Скрипт для проверки качества датасета
Проверяет: размеры, битые файлы, дубликаты, распределение классов
"""

import os
from PIL import Image
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

class DatasetChecker:
    """Класс для проверки качества датасета"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_sizes = []
        self.corrupted_images = []
        self.image_hashes = defaultdict(list)
        self.class_distribution = {}

    def check_all(self):
        """Запуск всех проверок"""
        print("=" * 80)
        print("🔍 ПОЛНАЯ ПРОВЕРКА КАЧЕСТВА ДАТАСЕТА")
        print("=" * 80)
        print(f"📁 Путь: {self.dataset_path}\n")

        self._check_structure()
        self._check_images()
        self._check_duplicates()
        self._print_summary()

    def _check_structure(self):
        """Проверка структуры датасета"""
        print("📂 Проверка структуры датасета...")

        classes = sorted([d for d in os.listdir(self.dataset_path)
                         if os.path.isdir(os.path.join(self.dataset_path, d))])

        print(f"✅ Найдено классов: {len(classes)}")

        for cls in classes:
            cls_path = os.path.join(self.dataset_path, cls)
            images = [f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            self.class_distribution[cls] = len(images)
            print(f"  📂 {cls}: {len(images)} изображений")

        # Проверка баланса классов
        counts = list(self.class_distribution.values())
        if len(set(counts)) > 1:
            print(f"\n⚠️  ВНИМАНИЕ: Классы несбалансированы!")
            print(f"   Мин: {min(counts)}, Макс: {max(counts)}, Разница: {max(counts) - min(counts)}")
        else:
            print(f"\n✅ Классы сбалансированы: по {counts[0]} изображений")

        print()

    def _check_images(self):
        """Проверка всех изображений"""
        print("🖼️  Проверка изображений...")

        total_images = 0

        for class_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue

                img_path = os.path.join(class_path, img_name)
                total_images += 1

                try:
                    # Открываем изображение
                    img = Image.open(img_path)

                    # Проверяем что оно действительно читается
                    img.verify()

                    # Переоткрываем для получения размера (verify закрывает файл)
                    img = Image.open(img_path)
                    width, height = img.size

                    # Сохраняем размер
                    self.image_sizes.append((width, height))

                    # Вычисляем хеш для поиска дубликатов
                    with open(img_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        self.image_hashes[file_hash].append(img_path)

                except Exception as e:
                    self.corrupted_images.append((img_path, str(e)))

        print(f"✅ Проверено изображений: {total_images}")

        if self.corrupted_images:
            print(f"\n❌ Битых изображений: {len(self.corrupted_images)}")
            for img_path, error in self.corrupted_images[:5]:
                print(f"   {img_path}: {error}")
            if len(self.corrupted_images) > 5:
                print(f"   ... и еще {len(self.corrupted_images) - 5}")
        else:
            print("✅ Битых изображений не найдено")

        print()

    def _check_duplicates(self):
        """Проверка дубликатов"""
        print("🔄 Проверка дубликатов...")

        duplicates = {h: paths for h, paths in self.image_hashes.items() if len(paths) > 1}

        if duplicates:
            print(f"⚠️  Найдено дубликатов: {len(duplicates)} групп")
            for i, (file_hash, paths) in enumerate(list(duplicates.items())[:3]):
                print(f"\n  Группа {i+1} ({len(paths)} файлов):")
                for path in paths[:3]:
                    print(f"    - {path}")
                if len(paths) > 3:
                    print(f"    ... и еще {len(paths) - 3}")

            if len(duplicates) > 3:
                print(f"\n  ... и еще {len(duplicates) - 3} групп дубликатов")
        else:
            print("✅ Дубликатов не найдено")

        print()

    def _print_summary(self):
        """Печать итоговой статистики"""
        print("=" * 80)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 80)

        # Размеры изображений
        print("\n📐 Размеры изображений:")
        size_counter = Counter(self.image_sizes)

        for size, count in size_counter.most_common(15):
            is_square = "✅" if size[0] == size[1] else "❌"
            print(f"  {is_square} {size[0]:4d}×{size[1]:4d}: {count:4d} изображений")

        # Статистика квадратности
        square_images = sum(1 for w, h in self.image_sizes if w == h)
        total_images = len(self.image_sizes)

        print(f"\n📊 Квадратных изображений: {square_images}/{total_images} ({100*square_images/total_images:.1f}%)")

        if square_images < total_images:
            print("⚠️  РЕКОМЕНДАЦИЯ: Используйте SquarePad или Resize+CenterCrop")
        else:
            print("✅ Все изображения квадратные")

        # Диапазон размеров
        if self.image_sizes:
            widths = [w for w, h in self.image_sizes]
            heights = [h for w, h in self.image_sizes]

            print(f"\n📏 Диапазон размеров:")
            print(f"  Ширина:  {min(widths)} - {max(widths)} px")
            print(f"  Высота:  {min(heights)} - {max(heights)} px")
            print(f"  Средний: {sum(widths)//len(widths)}×{sum(heights)//len(heights)} px")

        # Проблемы
        print(f"\n🚨 ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ:")
        issues_found = False

        if self.corrupted_images:
            print(f"  ❌ Битых изображений: {len(self.corrupted_images)}")
            issues_found = True

        duplicates_count = sum(1 for paths in self.image_hashes.values() if len(paths) > 1)
        if duplicates_count > 0:
            print(f"  ⚠️  Групп дубликатов: {duplicates_count}")
            issues_found = True

        if square_images < total_images:
            print(f"  ⚠️  Неквадратных изображений: {total_images - square_images}")
            issues_found = True

        # Проверка разных размеров
        if len(size_counter) > 10:
            print(f"  ⚠️  Слишком много разных размеров: {len(size_counter)}")
            issues_found = True

        if not issues_found:
            print("  ✅ Проблем не обнаружено!")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Использование (замените на ваш путь)
    # dataset_path = "/content/drive/MyDrive/dataset_nik"

    # Для теста в локальной среде используем placeholder
    import sys

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        print("Usage: python check_dataset_quality.py <path_to_dataset>")
        print("Example: python check_dataset_quality.py /content/drive/MyDrive/dataset_nik")
        sys.exit(1)

    checker = DatasetChecker(dataset_path)
    checker.check_all()
