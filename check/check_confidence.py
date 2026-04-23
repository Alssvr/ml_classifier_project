# check_confidence.py
import pandas as pd
from pathlib import Path

# Найдите последний файл classified_
files = list(Path('data/processed').glob('classified_*.xlsx'))
latest = max(files, key=lambda p: p.stat().st_mtime)

df = pd.read_excel(latest)

print("="*50)
print("АНАЛИЗ УВЕРЕННОСТИ ПРЕДСКАЗАНИЙ")
print("="*50)

print(f"\nВсего записей: {len(df):,}")
print(f"\nРаспределение confidence:")
print(f"  Min: {df['confidence'].min():.4f}")
print(f"  Max: {df['confidence'].max():.4f}")
print(f"  Mean: {df['confidence'].mean():.4f}")
print(f"  Median: {df['confidence'].median():.4f}")

print(f"\nПроцентили:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = df['confidence'].quantile(p/100)
    print(f"  {p}%: {val:.4f}")

print(f"\nДиапазоны уверенности:")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(bins)-1):
    count = ((df['confidence'] >= bins[i]) & (df['confidence'] < bins[i+1])).sum()
    pct = count / len(df) * 100
    print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:,} ({pct:.1f}%)")

# Показываем примеры с разной уверенностью
print("\n" + "="*50)
print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
print("="*50)

print("\n🔴 Низкая уверенность (<0.3):")
low = df[df['confidence'] < 0.3].head(5)
for _, row in low.iterrows():
    print(f"  [{row['confidence']:.3f}] {row['predicted_class']}: {row['Наименование'][:50]}...")

print("\n🟡 Средняя уверенность (0.3-0.7):")
mid = df[(df['confidence'] >= 0.3) & (df['confidence'] < 0.7)].head(5)
for _, row in mid.iterrows():
    print(f"  [{row['confidence']:.3f}] {row['predicted_class']}: {row['Наименование'][:50]}...")

print("\n🟢 Высокая уверенность (>0.7):")
high = df[df['confidence'] > 0.7].head(5)
if len(high) > 0:
    for _, row in high.iterrows():
        print(f"  [{row['confidence']:.3f}] {row['predicted_class']}: {row['Наименование'][:50]}...")
else:
    print("  Нет записей с уверенностью >0.7")