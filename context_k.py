import re
import string
import numpy as np
from collections import defaultdict
from typing import List, Set, Dict, Tuple
from gensim.models import Word2Vec


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return [t for t in text.split() if t]


def kgrams(term: str, k: int = 2, use_boundaries: bool = False) -> Set[str]:
    t = ('^' + term + '$') if use_boundaries else term
    if len(t) < k:
        return {t}
    return {t[i:i+k] for i in range(len(t)-k+1)}


def build_kgram_index(documents: List[str], k: int = 2, use_boundaries: bool = False) -> Tuple[Dict[str, Set[str]], Set[str]]:
    vocab = set()
    for doc in documents:
        vocab.update(tokenize(doc))
    
    index = defaultdict(set)
    for term in vocab:
        for kg in kgrams(term, k, use_boundaries):
            index[kg].add(term)
    
    return dict(index), vocab


def generate_candidates_from_kgrams(query_word: str, index: Dict[str, Set[str]], 
                                    k: int = 2, use_boundaries: bool = False, 
                                    max_candidates: int = 1000) -> Set[str]:
    q_k = kgrams(query_word, k, use_boundaries)
    candidates = set()
    for kg in q_k:
        candidates.update(index.get(kg, ()))
        if len(candidates) >= max_candidates:
            break
    candidates.discard(query_word)
    return candidates


def train_word2vec_model(sentences: List[List[str]], vector_size: int = 100, 
                        window: int = 5, epochs: int = 50) -> Word2Vec:
    print(f"Training Word2Vec on {len(sentences)} sentences...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=0,
        workers=4,
        epochs=epochs,
        negative=5
    )
    print(f"✓ Model trained. Vocabulary: {len(model.wv)} words")
    return model


def get_context_vector(model: Word2Vec, context_words: List[str]) -> np.ndarray:
    valid_context = [w for w in context_words if w in model.wv]
    
    if not valid_context:
        return np.zeros(model.vector_size)
    
    vectors = np.array([model.wv[w] for w in valid_context])
    return np.mean(vectors, axis=0)


def score_candidates(model: Word2Vec, context_words: List[str], 
                    candidates: Set[str]) -> Dict[str, float]:
    valid_candidates = [c for c in candidates if c in model.wv]
    
    if not valid_candidates:
        return {}
    
    context_vec = get_context_vector(model, context_words)
    
    if np.allclose(context_vec, 0):
        return {c: 0.0 for c in valid_candidates}
    
    scores = {}
    for candidate in valid_candidates:
        cand_vec = model.wv[candidate]
        similarity = np.dot(context_vec, cand_vec) / (
            np.linalg.norm(context_vec) * np.linalg.norm(cand_vec) + 1e-10
        )
        scores[candidate] = float(similarity)
    
    return scores


def correct_spelling(sentence: str, model: Word2Vec, kgram_index: Dict[str, Set[str]], 
                    window_size: int = 2, k: int = 2, verbose: bool = False) -> str:
    words = tokenize(sentence)
    vocab = set(model.wv.index_to_key)
    corrected = []
    
    for i, word in enumerate(words):
        if word in vocab:
            corrected.append(word)
            if verbose:
                print(f"✓ '{word}' is correct")
        else:
            if verbose:
                print(f"\n✗ '{word}' needs correction")
            
            candidates = generate_candidates_from_kgrams(word, kgram_index, k=k)
            
            if not candidates:
                corrected.append(word)
                if verbose:
                    print(f"  No candidates found, keeping '{word}'")
                continue
            
            if verbose:
                print(f"  K-gram candidates ({len(candidates)}): {list(candidates)[:10]}")
            
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            context = [words[j] for j in range(start, end) if j != i]
            
            if verbose:
                print(f"  Context: {context}")
            
            scores = score_candidates(model, context, candidates)
            
            if not scores:
                corrected.append(word)
                if verbose:
                    print(f"  No scoreable candidates, keeping '{word}'")
                continue
            
            best = max(scores.items(), key=lambda x: x[1])
            
            if verbose:
                top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top candidates: {top_5}")
                print(f"  → Selected: '{best[0]}' (score: {best[1]:.4f})")
            
            corrected.append(best[0])
    
    return ' '.join(corrected)


def main():
    corpus = [
        "I love the smell of fresh flowers in the morning",
        "The perfume smells absolutely amazing and wonderful",
        "She enjoys smelling different types of flowers",
        "The spelling of this word is correct",
        "I need to check the spelling in my essay",
        "The weather is beautiful today",
        "We went to the beach and enjoyed the weather",
        "The cat is sleeping on the couch",
        "I enjoy reading books in my free time",
        "The coffee tastes really good",
        "She loves listening to music",
        "The sunset looks beautiful from here",
        "He is writing a letter to his friend",
        "The movie was very entertaining",
        "I like to go running in the park",
        "The food at this restaurant is delicious",
        "She is learning to play the guitar",
        "The garden is full of colorful flowers",
        "He enjoys painting in his studio",
        "The concert was absolutely fantastic"
    ]
    
    print("=" * 70)
    print("INTEGRATED SPELLING CORRECTION: K-GRAM + WORD2VEC")
    print("=" * 70)
    
    print("\n1. Preprocessing corpus...")
    sentences = [tokenize(doc) for doc in corpus]
    
    print("\n2. Building k-gram index...")
    kgram_index, vocab = build_kgram_index(corpus, k=2, use_boundaries=True)
    print(f"   K-gram index built with {len(kgram_index)} k-grams")
    print(f"   Vocabulary size: {len(vocab)}")
    
    print("\n3. Training Word2Vec model...")
    model = train_word2vec_model(sentences, vector_size=100, window=5, epochs=50)
    
    print("\n4. Testing integrated spelling correction...")
    print("=" * 70)
    
    test_cases = [
        "I love the speling of this word",
        "The perfume smeels amazing",
        "She enjoys smeling flowers",
        "The wether is beautiful today",
        "I like to go runing in the park",
        "The cofee tastes really good"
    ]
    
    for test in test_cases:
        print(f"\nInput:     {test}")
        corrected = correct_spelling(test, model, kgram_index, window_size=2, k=2, verbose=False)
        print(f"Corrected: {corrected}")
    
    print("\n\n5. Detailed example with verbose output...")
    print("=" * 70)
    example = "I love the speling and smeeling of this word"
    print(f"Input: {example}\n")
    corrected = correct_spelling(example, model, kgram_index, window_size=2, k=2, verbose=True)
    print(f"\nFinal output: {corrected}")
    
    print("\n" + "=" * 70)
    print("✓ Done!")


if __name__ == "__main__":
    main()
