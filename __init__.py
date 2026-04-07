"""
CritiqueConnect - A platform for enhancing creative feedback using AI models.

Hybrid Architecture:
  Stage 1: Preprocessing (preprocessor.py)
  Stage 2: BERT Semantic Analysis (analyzer.py)
  Stage 3: Gemini Generative Synthesis (enhancer.py, synthesizer.py)
  Stage 4: Final Report (synthesizer.py + main.py)
"""

from .collector import FeedbackCollector
from .preprocessor import FeedbackPreprocessor
from .analyzer import BERTAnalyzer
from .enhancer import GPTEnhancer
from .synthesizer import Synthesizer

__all__ = [
    'FeedbackCollector',
    'FeedbackPreprocessor',
    'BERTAnalyzer',
    'GPTEnhancer',
    'Synthesizer',
]