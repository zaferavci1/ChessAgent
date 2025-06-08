# Satranç Zekası Projesi

Bu proje, PyTorch kullanılarak geliştirilmiş bir Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning) satranç ajanıdır. Ajan, Q-learning algoritması ile eğitilmiş ve satranç pozisyonlarını değerlendirip hamle kararları vermek için bir Derin Q-Ağı (DQN) kullanmaktadır. Eğitim sürecinde, ajanın daha etkili öğrenmesi için Stockfish satranç motorundan rehber olarak yararlanılmıştır.

## Özellikler

- **Pekiştirmeli Öğrenme**: Ajan, en iyi hamleleri bulmayı Q-learning ile öğrenir.
- **Derin Sinir Ağı (DQN)**: Satranç tahtası durumunu değerlendirmek için evrişimli sinir ağları (CNN) tabanlı bir model kullanır.
- **Stockfish Entegrasyonu**: Eğitimde "öğretmen" rolü üstlenen Stockfish, ajanın daha hızlı ve stratejik öğrenmesine yardımcı olur.
- **Bulmaca Çözücü**: Eğitilmiş model, belirli satranç bulmacalarını (mat problemleri) çözmek için kullanılabilir.
- **Modüler Tasarım**: Model, eğitim ve uygulama kodları ayrı dosyalarda düzenlenmiştir.

## Nasıl Çalışır?

### Durum Temsili

Satranç tahtası, modelin anlayabileceği bir formata dönüştürülür. Bu projede, tahta durumu 16 katmanlı bir "bitboard" yapısı ile temsil edilir. Her katman 8x8'lik bir matristir ve tahtanın farklı bir özelliğini ifade eder:

- **12 katman**: Beyaz ve siyah taşların (Piyon, At, Fil, Kale, Vezir, Şah) konumları.
- **1 katman**: Boş kareler.
- **1 katman**: Sıradaki oyuncu (beyaz veya siyah).
- **1 katman**: Rok hakları.
- **1 katman**: Geçerken alma (en passant) hakkı.

Bu 16x8x8'lik yapı, sinir ağına girdi olarak verilir.

### Model Mimarisi

Model, bir Derin Q-Ağı'dır (DQN) ve aşağıdaki katmanlardan oluşur:

1.  **3 Evrişimli Katman (Convolutional Layers)**: Tahtadaki desenleri ve mekansal ilişkileri öğrenir.
2.  **2 Tam Bağlantılı Katman (Fully Connected Layers)**: Öğrenilen özelliklerden yola çıkarak hamleler için Q-değerlerini hesaplar.
3.  **Maskeleme Katmanı**: Yalnızca yasal hamlelerin değerlendirilmesini sağlamak için geçersiz hamlelerin Q-değerlerini sıfırlar.

Modelin çıktısı, 4096 olası hamlenin her biri için bir Q-değeridir.

### Eğitim

Ajan, `chess_agent_training.py` betiği ile eğitilir. Eğitim süreci şu adımları içerir:

- **Q-Learning**: Ajan, kendi kendine karşı (veya rastgele hamle yapan bir rakibe karşı) oyunlar oynar.
- **Ödül Mekanizması**: Hamlelerin ödülü, Stockfish'in pozisyon değerlendirmesindeki değişime göre belirlenir. Bu, ajanın Stockfish gibi oynamayı öğrenmesini teşvik eder.
- **Epsilon-Greedy Stratejisi**: Ajan, bazen en iyi bildiği hamleyi yapar (exploitation), bazen de yeni hamleler dener (exploration). Bu denge, `epsilon` değeri ile kontrol edilir.
- **Deneyim Tekrarı (Experience Replay)**: Ajanın yaptığı hamleler bir hafızada saklanır ve eğitim sırasında bu hafızadan rastgele örnekler seçilerek modelin daha stabil öğrenmesi sağlanır.

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekir:

- `torch`
- `chess`
- `pandas`
- `matplotlib`
- `numpy`

Bu kütüphaneleri `pip` ile yükleyebilirsiniz:
```bash
pip install torch chess pandas matplotlib numpy
```

Ayrıca, [Stockfish satranç motorunun](https://stockfishchess.org/download/) indirilmesi ve sisteminizde çalışır durumda olması gerekmektedir.

## Kullanım

### Yapılandırma

Eğitim betiğini çalıştırmadan önce, `chess_agent_training.py` dosyasındaki Stockfish motorunun yolunu kendi sisteminize göre güncellemeniz gerekmektedir.

```python
# chess_agent_training.py

def train():
    # ...
    # Kullanıcının sistemine göre stockfish yolunu ayarlayın
    stockfish_path = "/path/to/your/stockfish" # <--- BU YOLU DEĞİŞTİRİN
    # ...
```

### Eğitimi Başlatma

Modeli eğitmek için aşağıdaki komutu çalıştırın:

```bash
python chess_agent_training.py
```

Eğitim tamamlandığında, `chess_agent_50_games.pth` adında bir model dosyası oluşturulacaktır.

### Bulmaca Çözme

Eğitilmiş modeli kullanarak `main.py` içindeki satranç bulmacalarını çözmek için aşağıdaki komutu çalıştırın:

```bash
python main.py
```

Bu betik, eğitilmiş modeli yükler ve önceden tanımlanmış mat problemlerini çözmeye çalışır, ardından ajanın hamlesinin doğruluğunu kontrol eder. 