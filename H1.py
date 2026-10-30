
# -----------------------VERİ-----------------------------
# import numpy as np
# import matplotlib.pyplot as plt

# # 1. Tekrarlanabilirlik için rastgelelik tohumu (seed) belirleyelim
# np.random.seed(42)
# n_samples = 200

# # 2. İki farklı sınıf için 2 boyutlu normal dağılıma sahip veriler oluşturalım
# # Sınıf 0 (Etiket 0) - Merkez noktası (2, 2) civarında
# class_0_x = np.random.normal(loc=2.0, scale=0.8, size=(n_samples // 2, 2))
# y_0 = np.zeros(n_samples // 2)

# # Sınıf 1 (Etiket 1) - Merkez noktası (6, 6) civarında
# class_1_x = np.random.normal(loc=6.0, scale=0.8, size=(n_samples // 2, 2))
# y_1 = np.ones(n_samples // 2)

# # Verileri ve etiketleri birleştirelim
# X = np.vstack((class_0_x, class_1_x))
# y = np.hstack((y_0, y_1))

# # 3. Veriyi görselleştirelim
# plt.figure(figsize=(8, 6))
# plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Sınıf 0', alpha=0.7, edgecolor='k')
# plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Sınıf 1', alpha=0.7, edgecolor='k')

# # İki sınıfı ayıran örnek bir karar sınırı (Decision Boundary) çizgisi ekleyelim
# # Örneğin: x2 = -x1 + 8 doğrusu bu iki kümeyi net bir şekilde ayırır.
# x_line = np.linspace(0, 8, 100)
# y_line = -1.0 * x_line + 8.0
# plt.plot(x_line, y_line, color='green', linestyle='--', linewidth=2, label='Karar Sınırı (Perceptron Doğrusu)')

# plt.title('Tek Katmanlı Perceptron için Doğrusal Ayrılabilir 2D Veri Seti')
# plt.xlabel('Öznitelik 1 (X1)')
# plt.ylabel('Öznitelik 2 (X2)')
# plt.legend()
# plt.grid(True)
# plt.show()





# import numpy as np
# import matplotlib.pyplot as plt

# # =====================================================
# # 1. VERİ SETİ OLUŞTURMA
# # =====================================================
# np.random.seed(42)
# n_samples = 200

# X_0, X_1 = [], []

# # Verileri geniş bir alana (0 ile 10 arası) rastgele serpiştirelim
# while len(X_0) < n_samples // 2:
#     x1, x2 = np.random.uniform(0, 10, 2)
#     if x2 < x1 - 1.0:
#         X_0.append([x1, x2])

# while len(X_1) < n_samples // 2:
#     x1, x2 = np.random.uniform(0, 10, 2)
#     if x2 > x1 + 1.0:
#         X_1.append([x1, x2])

# X_0 = np.array(X_0)
# X_1 = np.array(X_1)

# # Verileri ve etiketleri birleştirelim
# X = np.vstack((X_0, X_1))
# y = np.hstack((np.zeros(len(X_0)), np.ones(len(X_1))))

# print("X (Veri) boyutu:", X.shape)
# print("y (Etiket) boyutu:", y.shape)

# # =====================================================
# # 2. PERCEPTRON PARAMETRELERİ
# # =====================================================
# learning_rate = 0.1     # verilen
# threshold     = 0.03    # verilen (basamak fonksiyonunun eşik değeri)
# max_epochs    = 1000    # serbest seçilen: üst sınır
# rng = np.random.default_rng(42)

# # Ağırlıklar ve bias: küçük rastgele başlangıç değerleri
# w = rng.uniform(-0.5, 0.5, size=2)
# b = rng.uniform(-0.5, 0.5)


# # =====================================================
# # 3. BASAMAK AKTİVASYON FONKSİYONU
# # =====================================================
# def step(net, thr=threshold):
#     return 1 if net >= thr else 0


# def predict(X, w, b):
#     net = X @ w + b
#     return (net >= threshold).astype(int)


# # =====================================================
# # 4. EĞİTİM
# # =====================================================
# errors_per_epoch = []

# for epoch in range(max_epochs):
#     # Her epoch'ta örnekleri karıştır
#     idx = rng.permutation(len(X))
#     total_errors = 0

#     for i in idx:
#         net = np.dot(w, X[i]) + b
#         y_hat = step(net)
#         error = y[i] - y_hat          # d - y

#         if error != 0:
#             # Perceptron öğrenme kuralı
#             w = w + learning_rate * error * X[i]
#             b = b + learning_rate * error
#             total_errors += 1

#     errors_per_epoch.append(total_errors)

#     # Hiç hata yoksa eğitimi bitir
#     if total_errors == 0:
#         print(f"Eğitim {epoch + 1}. epoch'ta yakınsadı.")
#         break

# # =====================================================
# # 5. SONUÇLAR
# # =====================================================
# y_pred = predict(X, w, b)
# accuracy = np.mean(y_pred == y) * 100

# print("Ağırlıklar (w1, w2):", w)
# print("Bias (b):", b)
# print(f"Doğruluk: %{accuracy:.2f}")

# # =====================================================
# # 6. GÖRSELLEŞTİRME
# # =====================================================
# # Karar sınırı: w1*x1 + w2*x2 + b = threshold
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# # Veri ve karar sınırı
# ax[0].scatter(X_0[:, 0], X_0[:, 1], c="tab:blue", label="Sınıf 0")
# ax[0].scatter(X_1[:, 0], X_1[:, 1], c="tab:red", label="Sınıf 1")
# x_line = np.linspace(0, 10, 100)
# if abs(w[1]) > 1e-12:
#     x2_line = (threshold - b - w[0] * x_line) / w[1]
#     ax[0].plot(x_line, x2_line, "k--", label="Karar sınırı")
# ax[0].set_xlim(0, 10)
# ax[0].set_ylim(0, 10)
# ax[0].set_xlabel("x1")
# ax[0].set_ylabel("x2")
# ax[0].set_title("Perceptron Karar Sınırı")
# ax[0].legend()

# # Epoch başına hata sayısı
# ax[1].plot(range(1, len(errors_per_epoch) + 1), errors_per_epoch, marker="o")
# ax[1].set_xlabel("Epoch")
# ax[1].set_ylabel("Hatalı sınıflandırma sayısı")
# ax[1].set_title("Eğitim Hatası")

# plt.tight_layout()
# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt

# # =====================================================
# # 1. VERİ SETİ OLUŞTURMA
# # =====================================================
# np.random.seed(42)
# n_samples = 200

# X_0, X_1 = [], []

# while len(X_0) < n_samples // 2:
#     x1, x2 = np.random.uniform(0, 10, 2)
#     if x2 < x1 - 1.0:
#         X_0.append([x1, x2])

# while len(X_1) < n_samples // 2:
#     x1, x2 = np.random.uniform(0, 10, 2)
#     if x2 > x1 + 1.0:
#         X_1.append([x1, x2])

# X_0 = np.array(X_0)
# X_1 = np.array(X_1)

# X = np.vstack((X_0, X_1))
# y = np.hstack((np.zeros(len(X_0)), np.ones(len(X_1))))

# # =====================================================
# # 2. DEĞİŞTİRİLECEK PARAMETRELER 
# # =====================================================
# # Her satır: (learning_rate, eşik_değeri, grafikteki_etiket)
# # Birinci satır ödevde verilen referans ayardır.
# ayarlar = [
#     (0.1,  0.03, "Referans: lr=0.1, eşik=0.03"),
#     (0.01, 0.50, "Deney 1:  lr=0.01, eşik=0.5"),
#     (0.5, -0.50, "Deney 2:  lr=0.5, eşik=-0.5"),
#     (1.0,  2.00, "Deney 3:  lr=1.0, eşik=2.0"),
# ]

# # =====================================================
# # 3. SABİT KALAN PARAMETRELER
# # =====================================================
# max_epochs = 1000   # en fazla epoch sayısı
# seed       = 42     # başlangıç ağırlıkları ve karıştırma sırası için


# # =====================================================
# # 4. PERCEPTRON EĞİTİM FONKSİYONU
# # =====================================================
# def perceptron_egit(lr, esik):
#     # Her deneyde aynı tohum -> aynı başlangıç ağırlıkları ve aynı karıştırma sırası
#     rng = np.random.default_rng(seed)
#     w = rng.uniform(-0.5, 0.5, size=2)
#     b = rng.uniform(-0.5, 0.5)

#     son_epoch = max_epochs
#     for epoch in range(max_epochs):
#         hata_sayisi = 0
#         for i in rng.permutation(len(X)):
#             net = np.dot(w, X[i]) + b
#             y_hat = 1 if net >= esik else 0      # basamak fonksiyonu
#             hata = y[i] - y_hat
#             if hata != 0:
#                 w = w + lr * hata * X[i]
#                 b = b + lr * hata
#                 hata_sayisi += 1
#         if hata_sayisi == 0:
#             son_epoch = epoch + 1
#             break

#     tahmin = ((X @ w + b) >= esik).astype(int)
#     dogruluk = np.mean(tahmin == y) * 100
#     return w, b, dogruluk, son_epoch


# # =====================================================
# # 5. HER AYAR İÇİN EĞİT VE KARAR SINIRINI ÇİZ
# # =====================================================
# plt.figure(figsize=(8, 7))
# plt.scatter(X_0[:, 0], X_0[:, 1], c="tab:blue", s=15, label="Sınıf 0")
# plt.scatter(X_1[:, 0], X_1[:, 1], c="tab:red", s=15, label="Sınıf 1")

# x_cizgi = np.linspace(0, 10, 200)

# for lr, esik, etiket in ayarlar:
#     w, b, dogruluk, epoch = perceptron_egit(lr, esik)
#     print(f"{etiket:32s} -> doğruluk: %{dogruluk:.1f}, epoch: {epoch}, "
#           f"w=({w[0]:.3f}, {w[1]:.3f}), b={b:.3f}")

#     # Karar sınırı: w1*x1 + w2*x2 + b = eşik
#     if abs(w[1]) > 1e-12:
#         x2_cizgi = (esik - b - w[0] * x_cizgi) / w[1]
#         plt.plot(x_cizgi, x2_cizgi, linewidth=2, label=etiket)
#     else:
#         plt.axvline((esik - b) / w[0], linewidth=2, label=etiket)

# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.title("Farklı Learning Rate ve Eşik Değerleri İçin Karar Sınırları")
# plt.legend(fontsize=8)
# plt.grid(alpha=0.3)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. VERİ SETİ OLUŞTURMA (iki ağ için ortak)
# =====================================================
np.random.seed(42)
n_samples = 200

X_0, X_1 = [], []

while len(X_0) < n_samples // 2:
    x1, x2 = np.random.uniform(0, 10, 2)
    if x2 < x1 - 1.0:
        X_0.append([x1, x2])

while len(X_1) < n_samples // 2:
    x1, x2 = np.random.uniform(0, 10, 2)
    if x2 > x1 + 1.0:
        X_1.append([x1, x2])

X_0 = np.array(X_0)
X_1 = np.array(X_1)

X = np.vstack((X_0, X_1))
y = np.hstack((np.zeros(len(X_0)), np.ones(len(X_1))))

# Adaline (LMS) kararlı çalışsın diye girişler 0-1 aralığına ölçeklenir.
# Hedef değerler -1 / +1 yapılır (0 / 1 etiketleri sınıf kontrolü için saklanır).
Xn = X / 10.0
d = 2 * y - 1

# =====================================================
# 2. DEĞİŞTİRİLECEK PARAMETRELER  (SADECE BURAYI DÜZENLE)
# =====================================================

# --- PERCEPTRON: learning rate ve eşik değiştirilir ---
# Her satır: (learning_rate, eşik_değeri, grafikteki_etiket)
ayarlar_perceptron = [
    (0.1,  0.03, "Referans: lr=0.1, eşik=0.03"),
    (0.01, 0.50, "Deney 1:  lr=0.01, eşik=0.5"),
    (0.5, -0.50, "Deney 2:  lr=0.5, eşik=-0.5"),
    (1.0,  2.00, "Deney 3:  lr=1.0, eşik=2.0"),
]

# --- ADALINE DENEY A: learning rate ve eşik değiştirilir, minimum hata SABİT ---
min_hata_sabit = 0.25
# Her satır: (learning_rate, eşik_değeri, grafikteki_etiket)
ayarlar_adaline_A = [
    (0.1,   0.03, "Referans: lr=0.1, eşik=0.03"),
    (0.01,  0.50, "Deney 1:  lr=0.01, eşik=0.5"),
    (0.05, -0.30, "Deney 2:  lr=0.05, eşik=-0.3"),
    (0.3,   0.30, "Deney 3:  lr=0.3, eşik=0.3"),
]

# --- ADALINE DENEY B: sadece minimum hata değiştirilir, lr ve eşik SABİT ---
lr_sabit   = 0.01
esik_sabit = 0.03
# Her satır: (minimum_hata, grafikteki_etiket)
ayarlar_adaline_B = [
    (0.80, "min hata = 0.80"),
    (0.50, "min hata = 0.50"),
    (0.35, "min hata = 0.35"),
    (0.25, "min hata = 0.25"),
]

# =====================================================
# 3. SABİT KALAN PARAMETRELER
# =====================================================
max_epochs = 1000   # en fazla epoch sayısı
seed       = 42     # başlangıç ağırlıkları ve karıştırma sırası için


# =====================================================
# 4. EĞİTİM FONKSİYONLARI (hata geçmişini de döndürür)
# =====================================================
def perceptron_egit(lr, esik):
    # Her deneyde aynı tohum -> aynı başlangıç ağırlıkları ve aynı karıştırma sırası
    rng = np.random.default_rng(seed)
    w = rng.uniform(-0.5, 0.5, size=2)
    b = rng.uniform(-0.5, 0.5)

    hata_gecmisi = []   # her epoch'taki hatalı sınıflandırma sayısı
    for epoch in range(max_epochs):
        hata_sayisi = 0
        for i in rng.permutation(len(X)):
            net = np.dot(w, X[i]) + b
            y_hat = 1 if net >= esik else 0      # basamak fonksiyonu
            hata = y[i] - y_hat
            if hata != 0:
                w = w + lr * hata * X[i]
                b = b + lr * hata
                hata_sayisi += 1
        hata_gecmisi.append(hata_sayisi)
        if hata_sayisi == 0:
            break

    tahmin = ((X @ w + b) >= esik).astype(int)
    dogruluk = np.mean(tahmin == y) * 100
    return w, b, dogruluk, hata_gecmisi


def adaline_egit(lr, esik, min_hata):
    rng = np.random.default_rng(seed)
    w = rng.uniform(-0.5, 0.5, size=2)
    b = rng.uniform(-0.5, 0.5)

    hata_gecmisi = []   # her epoch sonundaki ortalama kare hata (MSE)
    for epoch in range(max_epochs):
        for i in rng.permutation(len(Xn)):
            net = np.dot(w, Xn[i]) + b       # doğrusal çıkış (eğitimde basamak YOK)
            hata = d[i] - net                # Widrow-Hoff (LMS) hatası
            w = w + lr * hata * Xn[i]
            b = b + lr * hata

        mse = np.mean((d - (Xn @ w + b)) ** 2)
        hata_gecmisi.append(mse)
        if mse <= min_hata:                  # öğrenmeyi sonlandırma koşulu
            break

    # Sınıflandırma: basamak fonksiyonu (net >= eşik -> 1, değilse 0)
    tahmin = ((Xn @ w + b) >= esik).astype(int)
    dogruluk = np.mean(tahmin == y) * 100

    # Ağırlıkları orijinal (ölçeklenmemiş) koordinatlara çevir
    return w / 10.0, b, dogruluk, hata_gecmisi


# =====================================================
# 5. ÇİZİM VE YAZDIRMA FONKSİYONLARI
# =====================================================
def sekil_ciz(baslik, sonuclar, hata_etiketi):
    """Sol: karar sınırları (üst üste). Sağ: eğitim sırasındaki hata eğrileri."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(baslik)

    # --- Sol grafik: veri + karar sınırları ---
    ax[0].scatter(X_0[:, 0], X_0[:, 1], c="tab:blue", s=15, label="Sınıf 0")
    ax[0].scatter(X_1[:, 0], X_1[:, 1], c="tab:red", s=15, label="Sınıf 1")
    x_cizgi = np.linspace(0, 10, 200)

    for s in sonuclar:
        w, b, esik = s["w"], s["b"], s["esik"]
        # Karar sınırı: w1*x1 + w2*x2 + b = eşik
        if abs(w[1]) > 1e-12:
            ax[0].plot(x_cizgi, (esik - b - w[0] * x_cizgi) / w[1],
                       linewidth=2, label=s["etiket"])
        else:
            ax[0].axvline((esik - b) / w[0], linewidth=2, label=s["etiket"])

    ax[0].set_xlim(0, 10)
    ax[0].set_ylim(0, 10)
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].set_title("Karar Sınırları")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # --- Sağ grafik: hata eğrileri ---
    for s in sonuclar:
        ax[1].plot(range(1, len(s["hata"]) + 1), s["hata"],
                   marker="o", markersize=3, label=s["etiket"])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel(hata_etiketi)
    ax[1].set_title("Eğitim Sırasındaki Hata Eğrisi")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    plt.tight_layout()


def sonuc_yazdir(baslik, sonuclar):
    print("=" * 78)
    print(baslik)
    print("=" * 78)
    for s in sonuclar:
        print(s["etiket"])
        print(f"   lr = {s['lr']}, eşik = {s['esik']}")
        print(f"   w1 = {s['w'][0]:.4f}, w2 = {s['w'][1]:.4f}, b = {s['b']:.4f}")
        print(f"   son hata = {s['hata'][-1]:.4f}, epoch = {len(s['hata'])}, "
              f"doğruluk = %{s['dogruluk']:.1f}")
    print()


# =====================================================
# 6. PERCEPTRON DENEYİ
# =====================================================
sonuclar_p = []
for lr, esik, etiket in ayarlar_perceptron:
    w, b, dogruluk, hata = perceptron_egit(lr, esik)
    sonuclar_p.append(dict(lr=lr, esik=esik, etiket=etiket,
                           w=w, b=b, dogruluk=dogruluk, hata=hata))

sonuc_yazdir("PERCEPTRON - son parametre değerleri (hata = hatalı örnek sayısı)",
             sonuclar_p)
sekil_ciz("Perceptron: Farklı Learning Rate ve Eşik Değerleri",
          sonuclar_p, "Hatalı sınıflandırma sayısı")

# =====================================================
# 7. ADALINE DENEY A: lr ve eşik değişiyor
# =====================================================
sonuclar_a = []
for lr, esik, etiket in ayarlar_adaline_A:
    w, b, dogruluk, hata = adaline_egit(lr, esik, min_hata_sabit)
    sonuclar_a.append(dict(lr=lr, esik=esik, etiket=etiket,
                           w=w, b=b, dogruluk=dogruluk, hata=hata))

sonuc_yazdir(f"ADALINE DENEY A (min hata = {min_hata_sabit}) - "
             "son parametre değerleri (hata = MSE)", sonuclar_a)
sekil_ciz("Adaline: Farklı Learning Rate ve Eşik Değerleri",
          sonuclar_a, "MSE (ortalama kare hata)")

# =====================================================
# 8. ADALINE DENEY B: sadece minimum hata değişiyor
# =====================================================
sonuclar_b = []
for min_hata, etiket in ayarlar_adaline_B:
    w, b, dogruluk, hata = adaline_egit(lr_sabit, esik_sabit, min_hata)
    sonuclar_b.append(dict(lr=lr_sabit, esik=esik_sabit, etiket=etiket,
                           w=w, b=b, dogruluk=dogruluk, hata=hata))

sonuc_yazdir(f"ADALINE DENEY B (lr = {lr_sabit}, eşik = {esik_sabit}) - "
             "son parametre değerleri (hata = MSE)", sonuclar_b)
sekil_ciz("Adaline: Farklı Minimum Hata Değerleri",
          sonuclar_b, "MSE (ortalama kare hata)")

plt.show()