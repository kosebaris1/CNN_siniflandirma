# Derin Öğrenme ile Banknot Sınıflandırma (TL vs USD)

Bu proje, **Türk Lirası (TL)** ve **Amerikan Doları (USD)** banknotlarını ayırt etmek amacıyla geliştirilmiş kapsamlı bir görüntü işleme ve derin öğrenme projesidir. Proje kapsamında üç farklı yaklaşım (Transfer Learning, Temel CNN, Optimize Edilmiş CNN) denenmiş, hiperparametre optimizasyonu ve veri artırma teknikleri ile en yüksek başarım hedeflenmiştir.


##  1. Veri Seti (Dataset)
**Özgünlük:** Projede kullanılan veri seti, internetten hazır alınmamış, **tamamen tarafımca çekilen** özgün görüntülerden oluşturulmuştur.

* **Sınıflar:** `TL` (Türk Lirası) ve `USD` (Amerikan Doları).
* **Veri Dağılımı:**
    * **Eğitim (Train):** %70
    * **Doğrulama (Validation):** %15
    * **Test (Test):** %15
* **Ön İşleme:** Tüm görüntüler `128x128` piksel boyutuna yeniden boyutlandırılmış ve `0-1` aralığında normalize edilmiştir.

---

## 2. Modeller ve Yöntemler

Model gelişimini bilimsel bir süreçle yönetmek adına 3 aşamalı bir strateji izlenmiştir:

###  Model 1: Transfer Learning (VGG16)
Literatürde başarısı kanıtlanmış **VGG16** mimarisi kullanılmıştır.
* **Yöntem:** ImageNet ağırlıkları kullanılarak "Feature Extraction" (Öznitelik Çıkarımı) yapılmıştır.
* **Konfigürasyon:** VGG16'nın taban katmanları dondurulmuş (`trainable=False`), çıkışına projeye özgü sınıflandırıcı (Flatten + Dense + Dropout) eklenmiştir.
* **Amaç:** Az veri ile yüksek başarım sağlayan state-of-the-art bir mimariyi test etmek.

###  Model 2: Temel CNN (Baseline)
Sıfırdan eğitilen, basit bir Evrişimli Sinir Ağı (CNN) modelidir.
* **Mimari:** 2 Bloklu Evrişim Katmanı (32 ve 64 Filtre).
* **Amaç:** Referans (Baseline) bir başarı skoru elde etmek ve Model 3'teki iyileştirmeler için zemin hazırlamak.
* **Sonuç:** %89.16 Test Doğruluğu.

###  Model 3: Geliştirilmiş CNN (Hiperparametre Optimizasyonu)
Model 2'nin performansını artırmak ve overfitting'i önlemek için **8 farklı kontrollü deney** yapılmıştır.
* **Kullanılan Teknikler:**
    * **Data Augmentation:** Veri çeşitliliğini artırmak için döndürme, kaydırma ve yakınlaştırma.
    * **Dropout:** Ezberlemeyi önlemek için nöron kapatma (%30 - %50 arası).
    * **Learning Rate Scheduling:** Hata minimumuna inmek için hassas hız ayarı (0.001 -> 0.0001).
    * **Derinlik Artışı:** 3 ve 4 katmanlı mimariler denenmiştir.

---

##  3. Deneysel Sonuçlar ve Performans Analizi

Model 3 geliştirilirken yapılan deneylerin özeti ve sonuç tablosu aşağıdadır. En iyi sonuç **Deney 4** ile elde edilmiştir.

| Deney | Mimari (Filtreler) | Veri Artırımı | Batch | LR | Dropout | Epoch | Test Doğruluğu | Yorum |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 3 Blok [32,64,128] | KAPALI | 32 | 0.001 | 0.3 | 20 | **%91.57** | Başlangıç referansı |
| **2** | 3 Blok [32,64,128] | KAPALI | 64 | 0.001 | 0.3 | 20 | **%92.77** | Batch 64 etkisi (Stabilite arttı) |
| **3** | 3 Blok [32,64,128] | KAPALI | 64 | 0.0002 | 0.4 | 20 | **%92.77** | Düşük LR ile hassas öğrenme |
| **4** | 4 Blok [32..256] | KAPALI | 64 | 0.0002 | 0.5 | 25 | **%95.18** | **EN İYİ MODEL (Derin Mimari)** |
| **5** | 3 Blok [32,64,128] | AÇIK | 64 | 0.001 | 0.3 | 25 | **%80.72** | Yüksek hız, zorlu veride başarısız oldu |
| **6** | 3 Blok [32,64,128] | AÇIK | 64 | 0.0002 | 0.4 | 25 | **%84.34** | Hız düşürülünce toparlanma başladı |
| **7** | 4 Blok [32..256] | AÇIK | 64 | 0.0005 | 0.4 | 30 | **%86.75** | Derin mimari zorlu veriyi daha iyi öğrendi |

###  Final Karşılaştırma
Projenin sonunda elde edilen en iyi test doğruluk oranları:

* **Model 2 (Temel):** %89.16
* **Model 3 (Final):** **%95.18** 

> **Sonuç:** Yapılan optimizasyonlar sonucunda, kendi tasarladığımız **Model 3**, Temel Model'e (Model 2) göre **%5-6'lık bir performans artışı** sağlamış ve daha kararlı bir yapıya kavuşmuştur.

---

## 📈 Grafikler
## Model3 grafik
<img width="1035" height="369" alt="image" src="https://github.com/user-attachments/assets/b9a4c1f4-012d-4770-9044-ea5b829c79cc" />


---

## 🛠️ Kurulum ve Kullanım

Proje Google Colab üzerinde geliştirilmiştir. Çalıştırmak için:

1.  Repoyu klonlayın.
2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install tensorflow split-folders matplotlib
    ```
3.  `Model3.ipynb` dosyasını çalıştırarak final sonuçları gözlemleyebilirsiniz.
