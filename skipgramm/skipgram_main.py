import time
import numpy as np
from collections import Counter
from gensim.models import KeyedVectors
from tqdm import tqdm

# Skip-Gram model with full softmax
class SkipGramFull:
    def __init__(self, vocab_size, emb_dim):
        # Инициализация весов случайными значениями
        self.W = np.random.uniform(-0.8, 0.8, (vocab_size, emb_dim))
        self.W_context = np.random.uniform(-0.8, 0.8, (vocab_size, emb_dim))
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
    
    def forward(self, X):
        # X - индексы слов (batch_size,)
        # Получаем эмбеддинги для центральных слов
        word_emb = self.W[X]  # (batch_size, emb_dim)
        
        # Скалярное произведение с эмбеддингами контекстных слов
        scores = np.dot(word_emb, self.W_context.T)  # (batch_size, vocab_size)
        
        # Применяем softmax для получения вероятностей
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs
    
    def train(self, pairs, epochs=5, lr=0.1, batch_size=512, verbose=True):
        n_samples = len(pairs)
        indices = np.arange(n_samples)
        losses = []
        
        for epoch in tqdm(range(epochs), desc="Training Full Softmax"):
            # Перемешиваем данные
            np.random.shuffle(indices)
            epoch_loss = 0
            
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_pairs = [pairs[i] for i in batch_indices]
                
                # Разделяем центральные слова и контекстные
                center_words = np.array([pair[0] for pair in batch_pairs])
                context_words = np.array([pair[1] for pair in batch_pairs])
                
                # Прямой проход
                probs = self.forward(center_words)  # (batch_size, vocab_size)
                
                # Вычисляем loss
                batch_size = len(center_words)
                loss = -np.sum(np.log(probs[np.arange(batch_size), context_words])) / batch_size
                epoch_loss += loss
                
                # Обратный проход
                d_probs = probs.copy()
                d_probs[np.arange(batch_size), context_words] -= 1
                d_probs /= batch_size
                
                # Обновляем веса
                d_W_context = np.dot(self.W[center_words].T, d_probs).T
                d_W = np.dot(d_probs, self.W_context).T
                
                for i, idx in enumerate(center_words):
                    self.W[idx] -= lr * d_W[:, i].T
                
                self.W_context -= lr * d_W_context
            
            losses.append(epoch_loss / (n_samples // batch_size + 1))
            if verbose and epoch % 5 == 0:
                tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {losses[-1]:.4f}")
        
        return losses
    
    def get_word_vector(self, word_idx):
        return self.W[word_idx]

# Skip-Gram model with Negative Sampling
class SkipGramNS:
    def __init__(self, vocab_size, emb_dim, negative_samples=5, noise_dist=None):
        # Инициализация весов случайными значениями
        self.W = np.random.uniform(-0.8, 0.8, (vocab_size, emb_dim))
        self.W_context = np.random.uniform(-0.8, 0.8, (vocab_size, emb_dim))
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.negative_samples = negative_samples
        
        # Распределение шума для negative sampling
        if noise_dist is None:
            self.noise_dist = np.ones(vocab_size) / vocab_size
        else:
            self.noise_dist = noise_dist
            
        # Предварительное создание таблицы для быстрого семплирования
        self._create_sampling_table(table_size=100000)
    
    def _create_sampling_table(self, table_size=1000000):
        """Создаёт таблицу для быстрого семплирования негативных примеров"""
        # Создаем таблицу с весами в соответствии с шумовым распределением
        self.sampling_table = np.zeros(table_size, dtype=np.int32)
        
        # Заполняем индексы в соответствии с вероятностями
        p = 0
        i = 0
        # Преобразуем распределение вероятностей в кумулятивное
        cumulative_dist = np.cumsum(self.noise_dist)
        
        # Для каждой позиции в таблице определяем элемент на основе кумулятивного распределения
        while i < table_size:
            if p >= len(cumulative_dist) - 1:
                break
            if i / table_size > cumulative_dist[p]:
                p += 1
            self.sampling_table[i] = p
            i += 1
            
        # Дозаполняем оставшуюся часть таблицы последним индексом
        while i < table_size:
            self.sampling_table[i] = len(cumulative_dist) - 1
            i += 1
    
    def sample_negative_fast(self, context_word, n_samples):
        """Быстрое семплирование без проверки дубликатов"""
        negative_samples = []
        table_size = len(self.sampling_table)
        
        for _ in range(n_samples + 10):  # Берем с запасом, чтоб не было бесконечных циклов
            rand_idx = np.random.randint(0, table_size)
            neg_sample = self.sampling_table[rand_idx]
            if neg_sample != context_word and neg_sample not in negative_samples:
                negative_samples.append(neg_sample)
                if len(negative_samples) == n_samples:
                    break
        
        # Если не набрали нужное количество, берем любые
        while len(negative_samples) < n_samples:
            rand_idx = np.random.randint(0, self.vocab_size)
            if rand_idx != context_word and rand_idx not in negative_samples:
                negative_samples.append(rand_idx)
                
        return np.array(negative_samples)
    
    def sigmoid(self, x):
        """Векторизованная сигмоида с clipping для численной стабильности"""
        # Ограничиваем значения для избежания проблем с числами
        x = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-x))
    
    def forward_vectorized(self, center_words, context_words, negative_samples_batch):
        """Векторизованный forward pass для батча примеров"""
        batch_size = len(center_words)
        
        # Эмбеддинги центральных слов [batch_size, emb_dim]
        center_embs = self.W[center_words]
        
        # Эмбеддинги контекстных слов [batch_size, emb_dim]
        context_embs = self.W_context[context_words]
        
        # Скалярное произведение для положительных примеров [batch_size]
        pos_scores = np.sum(center_embs * context_embs, axis=1)
        pos_probs = self.sigmoid(pos_scores)
        
        # Вычисляем лосс от положительных примеров
        pos_loss = -np.sum(np.log(pos_probs))
        
        # Обрабатываем негативные примеры
        neg_loss = 0
        neg_grads = []
        
        for i in range(batch_size):
            # Получаем эмбеддинги для негативных примеров [n_samples, emb_dim]
            neg_embs = self.W_context[negative_samples_batch[i]]
            
            # Скалярное произведение [n_samples]
            neg_scores = np.dot(center_embs[i], neg_embs.T)
            
            # Вероятности НЕ контекста
            neg_probs = self.sigmoid(-neg_scores)
            
            # Суммируем лосс
            neg_loss += -np.sum(np.log(neg_probs))
            
            # Расчет градиентов для негативных примеров
            d_neg = 1 - neg_probs
            
            # Обновление весов негативных примеров
            for j, neg_idx in enumerate(negative_samples_batch[i]):
                self.W_context[neg_idx] -= self.lr * d_neg[j] * center_embs[i]
            
            # Градиент для центрального слова от негативных примеров
            d_center_emb_neg = np.zeros(self.emb_dim)
            for j, neg_idx in enumerate(negative_samples_batch[i]):
                d_center_emb_neg += d_neg[j] * self.W_context[neg_idx]
            
            neg_grads.append(d_center_emb_neg)
        
        # Общий лосс
        total_loss = (pos_loss + neg_loss) / batch_size
        
        return total_loss, pos_probs, center_embs, context_embs, neg_grads
    
    def train(self, pairs, epochs=5, lr=0.1, batch_size=128, verbose=True):
        n_samples = len(pairs)
        indices = np.arange(n_samples)
        losses = []
        self.lr = lr  # Сохраняем lr как атрибут класса для использования в forward_vectorized
        
        for epoch in tqdm(range(epochs), desc="Training Negative Sampling"):
            # Перемешиваем данные
            np.random.shuffle(indices)
            epoch_loss = 0
            
            # Процесс по батчам
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:min(start_idx + batch_size, n_samples)]
                batch_pairs = [pairs[i] for i in batch_indices]
                
                # Разделяем центральные слова и контекстные
                center_words = np.array([pair[0] for pair in batch_pairs])
                context_words = np.array([pair[1] for pair in batch_pairs])
                
                # Предварительно самплируем негативные примеры для всего батча
                negative_samples_batch = [
                    self.sample_negative_fast(context_words[i], self.negative_samples) 
                    for i in range(len(context_words))
                ]
                
                # Векторизованный forward pass
                batch_loss, pos_probs, center_embs, context_embs, neg_grads = self.forward_vectorized(
                    center_words, context_words, negative_samples_batch
                )
                
                epoch_loss += batch_loss
                
                # Обратный проход и обновление весов
                
                # Градиенты для положительных примеров
                d_pos = pos_probs - 1  # [batch_size]
                
                # Обновляем контекстные веса для положительных примеров
                for i in range(len(center_words)):
                    # Градиент для контекстного слова
                    d_context_emb = d_pos[i] * center_embs[i]
                    self.W_context[context_words[i]] -= lr * d_context_emb
                    
                    # Градиент для центрального слова от положительного примера
                    d_center_emb = d_pos[i] * context_embs[i]
                    
                    # Добавляем градиент от негативных примеров
                    d_center_emb_total = d_center_emb + neg_grads[i]
                    
                    # Обновляем центральное слово
                    self.W[center_words[i]] -= lr * d_center_emb_total
            
            # Нормализуем loss по количеству батчей
            losses.append(epoch_loss / (n_samples // batch_size + 1))
            if verbose and epoch % 5 == 0:
                tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {losses[-1]:.4f}")
        
        return losses
    
    def get_word_vector(self, word_idx):
        return self.W[word_idx]


def main():
    print("Загрузка данных из файла sentences.txt...")
    # Загружаем предложения из файла
    with open("sentences.txt", "r", encoding="utf-8") as f:
        raw_sentences = f.read().strip().split('\n')
    
    # Разбиваем предложения на слова
    sentences = [sentence.split() for sentence in raw_sentences]
    
    # Строим словарь
    vocab = list({w for s in sentences for w in s})
    word2idx = {w:i for i,w in enumerate(vocab)}
    idx2word = {i:w for w,i in word2idx.items()}
    
    print(f"Словарь содержит {len(vocab)} уникальных слов")

    # Генерация обучающих пар (center, context)
    window = 2
    pairs = []
    for s in sentences:
        idxs = [word2idx[w] for w in s]
        for i,wi in enumerate(idxs):
            for j in range(max(0,i-window), min(len(idxs), i+window+1)):
                if i!=j:
                    pairs.append((wi, idxs[j]))
    
    print(f"Сгенерировано {len(pairs)} обучающих пар")

    # Простая шумовая дистрибуция P_n(w) ∝ freq(w)^(3/4)
    print("Вычисление распределения частот слов...")
    counts = Counter([wi for wi,_ in pairs])
    freqs = np.array([counts.get(i, 0) for i in range(len(vocab))], dtype=float)
    noise_dist = freqs**0.75
    noise_dist /= noise_dist.sum()

    # Параметры обучения
    embedding_size = 100
    epochs = 20
    
    # 1) Skip-Gram полный softmax (Hierarchical Softmax)
    print("\n== Обучение Skip-Gram с полным softmax ==")
    sg_full = SkipGramFull(vocab_size=len(vocab), emb_dim=embedding_size)
    start = time.time()
    sg_full.train(pairs, epochs=epochs, lr=0.05, batch_size=64, verbose=True)
    t_full = time.time() - start

    # 2) Skip-Gram + Negative Sampling
    print("\n== Обучение Skip-Gram с Negative Sampling (оптимизированная версия) ==")
    sg_ns = SkipGramNS(vocab_size=len(vocab), emb_dim=embedding_size, negative_samples=8, noise_dist=noise_dist)
    start = time.time()
    sg_ns.train(pairs, epochs=epochs, lr=0.05, batch_size=64, verbose=True)
    t_ns = time.time() - start

    print("\n=== Результаты ===")
    print(f"Время обучения полного softmax: {t_full:.2f}s")
    print(f"Время обучения Negative Sampling: {t_ns:.2f}s")
    print(f"Negative Sampling быстрее в {t_full/t_ns:.2f} раз")

    # --- Проверим близость слов ---
    def nearest(model, word, topn=5):
        if word not in word2idx:
            print(f"Слово '{word}' не найдено в словаре")
            return []
            
        idx = word2idx[word]
        vec = model.W[idx]
        # косинусная близость
        sims = []
        for j in range(len(vocab)):
            sims.append((j, vec.dot(model.W[j]) / (np.linalg.norm(vec)*np.linalg.norm(model.W[j]))))
        sims = sorted(sims, key=lambda x:-x[1])
        return [idx2word[j] for j,_ in sims[1:topn+1]]  # исключаем саму вершину

    # Список слов для проверки семантической близости
    words_to_check = ["кот", "собака", "человек", "врач", "программист"]
    
    print("\n=== Семантическая близость слов ===")
    for word in words_to_check:
        if word in word2idx:
            print(f"\nСоседи слова '{word}' (полный softmax):", nearest(sg_full, word))
            print(f"Соседи слова '{word}' (NS):", nearest(sg_ns, word))


if __name__ == "__main__":
    main() 