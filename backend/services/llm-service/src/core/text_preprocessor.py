import re
from typing import Optional


class TextPreprocessor:
    """Text preprocessing class for Persian/Arabic text normalization"""
    
    def normalize_persian_chars(self, text: str) -> str:
        """Convert Arabic characters to Persian"""
        replacements = {
            "ي": "ی",
            "ك": "ک",
            "ة": "ه",
            "ؤ": "و",
            "إ": "ا",
            "أ": "ا"
        }
        
        for arabic, persian in replacements.items():
            text = text.replace(arabic, persian)
        
        return text
    
    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic/Persian diacritics"""
        arabic_diacritics = re.compile("""
                                     ّ    | # tashdid
                                     َ    | # fatha
                                     ً    | # tanwin fatha
                                     ُ    | # damma
                                     ٌ    | # tanwin damma
                                     ِ    | # kasra
                                     ٍ    | # tanwin kasra
                                     ْ    | # sukun
                                 """, re.VERBOSE)
        
        return re.sub(arabic_diacritics, '', text)
    
    def remove_keshide(self, text: str) -> str:
        """Remove Kashida (tatweel/ـ) characters"""
        return text.replace("ـ", "")
    
    def keep_persian(self, text: str) -> str:
        """Keep only Persian characters and spaces"""
        pattern = r'[^آ-ی\s]'
        return re.sub(pattern, ' ', text)
    
    # def remove_english_words(self, text: str) -> str:
    #     """Remove English words (separate words)"""
    #     return re.sub(r'\b[a-zA-Z]+\b', '', text)
    
    # def remove_english_letters(self, text: str) -> str:
    #     """Remove all English letters"""
    #     return re.sub(r'[a-zA-Z]', '', text)
    
    def normalize_numbers(self, text: str) -> str:
        """Convert Persian numbers to English numbers"""
        persian_numbers = "۰۱۲۳۴۵۶۷۸۹"
        english_numbers = "0123456789"
        table = str.maketrans(persian_numbers, english_numbers)
        return text.translate(table)
    
    def remove_numbers(self, text: str) -> str:
        """Remove all numbers"""
        return re.sub(r'\d+', '', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (multiple spaces to single)"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def replace_halfspace_with_space(self, text: str) -> str:
        """Replace half-space (zero-width non-joiner) with space"""
        return text.replace("\u200c", " ")
    
    def remove_duplicate_lines(self, text: str) -> str:
        """Remove duplicate lines from text"""
        lines = text.split("\n")
        unique = list(dict.fromkeys(lines))
        return "\n".join(unique)
    
    def remove_punctuation(self, text: str) -> str:
        """Remove all Persian and English punctuation"""
        puncts = r""")(.><؟/\"'،؛:;!{}\[\]\-—_+=*&^%$#@~`|«»…,"""
        pattern = "[" + re.escape(puncts) + "]"
        text = re.sub(pattern, " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)
    
    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps in sequence"""
        text = self.remove_emojis(text)
        text = self.normalize_persian_chars(text)
        text = self.remove_diacritics(text)
        text = self.remove_keshide(text)
        # text = self.remove_english_letters(text)
        text = self.normalize_numbers(text)
        text = self.normalize_whitespace(text)
        text = self.replace_halfspace_with_space(text)
        text = self.remove_duplicate_lines(text)
        text = self.remove_punctuation(text)
        # text = self.remove_numbers(text)
        return text
