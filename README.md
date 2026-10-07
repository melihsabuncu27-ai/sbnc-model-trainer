# SBNC Model Trainer

Otomatik model eğitimi.

## Nasıl Çalışır?

- Her ayın 1'i ve 22'sinde otomatik eğitim başlar
- Model eğitilir, test edilir
- Başarılıysa GitHub Release'e yüklenir
- APK yeni sürümü indirir

## Klasör Yapısı

- `.github/workflows/train.yml` → Otomatik eğitim
- `train.py` → Eğitim kodu
- `config.json` → Ayarlar
- `requirements.txt` → Kütüphaneler
- `latest.json` → APK'nın okuduğu
