# src/rules_classifier.py
import pandas as pd
from typing import Optional, Dict, Tuple
from pathlib import Path
import joblib

from src.utils import setup_logging

logger = setup_logging()


class RuleBasedClassifier:
    """Классификатор на основе правил (ключевых слов)"""
    
    def __init__(self):
        self.first_word_rules: Dict[str, str] = {}
        self.contains_rules: Dict[str, str] = {}
        self.exact_match_rules: Dict[str, str] = {}
        self.stats = {'first_word': 0, 'contains': 0, 'exact': 0, 'none': 0}
    
    def add_first_word_rule(self, first_word: str, target_class: str):
        """Добавить правило: если текст начинается с first_word → target_class"""
        self.first_word_rules[first_word.lower()] = target_class
    
    def add_contains_rule(self, keyword: str, target_class: str):
        """Добавить правило: если содержит keyword → target_class"""
        self.contains_rules[keyword.lower()] = target_class
    
    def add_exact_rule(self, text: str, target_class: str):
        """Добавить правило: точное совпадение → target_class"""
        self.exact_match_rules[text.lower()] = target_class
    
    def auto_extract_rules(self, df: pd.DataFrame,
                           text_col: str = 'Наименование',
                           class_col: str = 'Шаблон класса',
                           min_confidence: float = 0.95,
                           max_rules: int = 500):
        """Автоматически извлечь правила из обучающих данных"""
        
        df = df.copy()
        df['first_word'] = df[text_col].astype(str).str.split().str[0].str.lower()
        
        rules_added = 0
        
        for first_word, group in df.groupby('first_word'):
            if len(group) < 5:
                continue
            
            if first_word == '' or pd.isna(first_word):
                continue
            
            top_class = group[class_col].value_counts()
            top_class_name = top_class.index[0]
            confidence = top_class.iloc[0] / len(group)
            
            if confidence >= min_confidence:
                self.add_first_word_rule(first_word, top_class_name)
                rules_added += 1
                
                if rules_added >= max_rules:
                    break
        
        logger.info(f"Извлечено {rules_added} правил по первому слову")
        return self
    
    def predict(self, text: str) -> Tuple[Optional[str], str, float]:
        """
        Предсказание на основе правил.
        Возвращает: (класс или None, метод, уверенность)
        """
        if pd.isna(text) or text == '':
            return None, 'none', 0.0
        
        text_lower = text.lower().strip()
        
        # 1. Точное совпадение
        if text_lower in self.exact_match_rules:
            self.stats['exact'] += 1
            return self.exact_match_rules[text_lower], 'exact_match', 1.0
        
        # 2. По первому слову
        words = text_lower.split()
        if words:
            first_word = words[0]
            if first_word in self.first_word_rules:
                self.stats['first_word'] += 1
                return self.first_word_rules[first_word], 'first_word', 0.95
        
        # 3. По содержанию ключевого слова
        for keyword, target_class in self.contains_rules.items():
            if keyword in text_lower:
                self.stats['contains'] += 1
                return target_class, 'contains', 0.85
        
        self.stats['none'] += 1
        return None, 'none', 0.0
    
    def get_stats(self) -> Dict:
        total = sum(self.stats.values())
        return {
            **self.stats,
            'total': total,
            'coverage': (total - self.stats['none']) / total if total > 0 else 0,
            'rules_count': len(self.first_word_rules)
        }
    
    def save(self, path: Path):
        joblib.dump(self, path)
        logger.info(f"Правила сохранены: {path}")
    
    @staticmethod
    def load(path: Path) -> 'RuleBasedClassifier':
        return joblib.load(path)


class HybridClassifier:
    """Гибридный классификатор: правила + ML"""
    
    def __init__(self, ml_classifier, rule_classifier: RuleBasedClassifier):
        self.ml_classifier = ml_classifier
        self.rule_classifier = rule_classifier
    
    def predict_single(self, text: str, ml_confidence_threshold: float = 0.3) -> Dict:
        """Гибридное предсказание для одного текста"""
        
        # Проверяем правила
        rule_class, method, rule_conf = self.rule_classifier.predict(text)
        
        if rule_class is not None:
            return {
                'predicted_class': rule_class,
                'method': method,
                'confidence': rule_conf,
                'second_guess': '',
                'needs_review': False
            }
        
        # Правила не сработали — используем ML
        import pandas as pd
        ml_result = self.ml_classifier.predict_with_confidence(pd.Series([text])).iloc[0]
        
        needs_review = ml_result['confidence'] < ml_confidence_threshold
        
        return {
            'predicted_class': ml_result['predicted_class'] if not needs_review else 'NEEDS_REVIEW',
            'method': 'ml',
            'confidence': ml_result['confidence'],
            'second_guess': ml_result.get('second_guess', ''),
            'needs_review': needs_review
        }
    
    def predict_dataframe(self, df: pd.DataFrame,
                          text_column: str = 'text_processed',
                          ml_confidence_threshold: float = 0.3) -> pd.DataFrame:
        """Классификация всего DataFrame"""
        
        results = []
        rule_stats = {'first_word': 0, 'contains': 0, 'exact': 0, 'ml': 0}
        
        for idx, row in df.iterrows():
            text = row[text_column]
            result = self.predict_single(text, ml_confidence_threshold)
            
            results.append({
                'index': idx,
                'predicted_class': result['predicted_class'],
                'method': result['method'],
                'confidence': result['confidence'],
                'second_guess': result.get('second_guess', ''),
                'needs_review': result['needs_review']
            })
            
            method = result['method']
            if method in rule_stats:
                rule_stats[method] += 1
        
        result_df = pd.DataFrame(results)
        
        # Выводим статистику
        total = len(result_df)
        rule_covered = sum(v for k, v in rule_stats.items() if k != 'ml')
        
        logger.info(f"Гибридная классификация завершена:")
        logger.info(f"  - Правилами покрыто: {rule_covered} ({rule_covered/total*100:.1f}%)")
        logger.info(f"  - ML: {rule_stats.get('ml', 0)} ({rule_stats.get('ml', 0)/total*100:.1f}%)")
        logger.info(f"  - Требуют проверки: {result_df['needs_review'].sum()}")
        
        return result_df