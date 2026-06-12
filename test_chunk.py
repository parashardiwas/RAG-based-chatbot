import re
from typing import Any

class FileParser:
    CHUNK_SIZE_WORDS = 250
    
    def _get_source_type(self, ext): return "text"
    
    def _split_into_topics(self, text):
        return [("Intro", text)]
        
    def _chunk_text(self, text: str, filename: str, file_ext: str) -> list[dict[str, Any]]:
        chunks = []
        source_type = self._get_source_type(file_ext)
        topics = self._split_into_topics(text)
        
        sentence_end_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
        for topic_title, topic_content in topics:
            sentences = sentence_end_pattern.split(topic_content)
            if not sentences: continue
                
            current_chunk_sentences = []
            current_word_count = 0
            
            for sentence in sentences:
                words = sentence.split()
                if not words: continue
                    
                current_chunk_sentences.append(sentence)
                current_word_count += len(words)
                
                if current_word_count >= self.CHUNK_SIZE_WORDS:
                    chunks.append({
                        "content": " ".join(current_chunk_sentences),
                    })
                    current_chunk_sentences = current_chunk_sentences[-1:]
                    current_word_count = len(current_chunk_sentences[0].split())
            
            if current_chunk_sentences and current_word_count > 0:
                # To prevent duplicating the last chunk if it was exactly CHUNK_SIZE_WORDS
                if not chunks or " ".join(current_chunk_sentences) != chunks[-1]["content"]:
                    chunks.append({
                        "content": " ".join(current_chunk_sentences),
                    })
        return chunks

p = FileParser()
text = "Hello world. " * 300
print(len(p._chunk_text(text, "test.txt", ".txt")))

text2 = "A" * 1000  # no sentences
print(len(p._chunk_text(text2, "test.txt", ".txt")))

