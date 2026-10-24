# YouTube Shorts Generator

Python kullanarak YouTube Shorts formatında (dikey, 9:16) videolar otomatik olarak üreten bir sistem.

## Özellikler

- **AI Senaryo Üretimi**: Gemini (gemini-3.5-flash-lite) ile konu başlığından 5-8 sahne senaryosu
- **Sahne Görselleri**: Gemini ile sahne başına dikey görsel üretimi + onay/değiştir akışı
- **Seslendirme**: Edge TTS ile Türkçe seslendirme
- **Altyazı**: Ses dosyasına senkronize altyazı
- **Video Montaj**: MoviePy ile 1080x1920 MP4 çıktı
- **Süre Kontrolü**: 40-60 saniye arasında video üretimi
- **Web Arayüzü (Gradio)**: Tarayıcıdan konu girip video üretme — canlı log, video önizleme, dışarıdan erişim (public link)
- **Workers AI Model Seçimi**: Web UI'da üç tekil dropdown ile Free uyumlu Text Generation, Text-to-Speech ve Text-to-Image modelleri arasından seçim; seçilen model aile bazlı payload adaptörleriyle Workers AI'ya gönderilir
- **Google Colab CLI**: Bulut tabanlı GPU/TPU ile uzaktan çalıştırma desteği

## Kurulum

### 1. Sanal Ortam

Proje `uv` ile yönetilir. Sanal ortam zaten mevcut olmalıdır:

```bash
source .venv/bin/activate  # Linux/Mac
```

### 2. Bağımlılıklar

```bash
uv pip install -r requirements.txt
```

### 3. Google Colab CLI (İsteğe Bağlı)

Uzaktan GPU/TPU ile çalıştırmak için:

```bash
uv tool install google-colab-cli
```

### 3. Cloudflare Girişi (API Anahtarı Olmadan)

Cloudflare Workers AI için `cf` CLI ile OAuth/device girişi yapabilirsiniz:

```bash
npm install --global cf
.venv/bin/python generate_shorts.py --cf-login
```

Komut doğrulama bağlantısını ve kodu gösterir; Cloudflare hesabınızla onayladıktan sonra senaryo, görsel ve Workers AI ses üretimi API anahtarı olmadan çalışır. Varsayılan Workers AI modelleri: metin `@cf/meta/llama-3.2-3b-instruct`, görsel `@cf/black-forest-labs/flux-1-schnell` (`steps=4`), ses `@cf/myshell-ai/melotts`. Melotts yalnızca `en/es/fr/jp/kr/zh` dillerini destekler; Türkçe video üretiminde varsayılan olarak Edge TTS kullanılır. Web UI açılışında `cf ai models list --per-page 100` kataloğundan `require_workers_paid` içermeyen modeller filtrelenir: **31 Text Generation, 10 Text-to-Image, 4 Text-to-Speech**. Üç dropdown seçimi pipeline'a açıkça taşınır; global `.env` model ayarları değiştirilmez. Erişim formu isteyen modeller etikette, LoRA modelleri adapter uyarısıyla gösterilir. Aura modelleri binary REST yanıtı, MeloTTS JSON/base64 yanıtı olarak desteklenir. Alternatif olarak `.env` içindeki `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN` REST yolu korunur.

### 4. API Anahtarları

`.env` dosyası oluşturun:

```bash
cp .env.example .env
```

`.env` dosyasını düzenleyin:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Kullanım

### Yerel Çalıştırma

```bash
.venv/bin/python generate_shorts.py "Konu başlığı"
```

Adım 2'de sahne başına **tek resim** üretilir ve **onay istenmez** (`IMAGE_CANDIDATES=1`, `APPROVE_IMAGES=false` — doğrudan kullanılır).
Onay akışı isterseniz `.env`'de `APPROVE_IMAGES=true` yapın: `Enter` = onayla, `d` = değiştir, `s` = atla, `q` = iptal (veya `--no-approve` ile her koşuda kapatın).

### Örnek

```bash
.venv/bin/python generate_shorts.py "Yapay zeka nedir?"
```

### Web Arayüzü (Gradio) — Dışarıdan Erişilebilir

```bash
# Varsayılan: 0.0.0.0:7860 dinler + *.gradio.live public link üretir
.venv/bin/python webui.py

# Sadece yerel/ağ erişimi (public link yok)
.venv/bin/python webui.py --no-share

# Farklı adres/port
.venv/bin/python webui.py --host 0.0.0.0 --port 8080
```

Arayüzde: **konu** metni, **ses motoru** seçimi (edge / cloudflare / ttsai), **canlı log** (5 adımın çıktısı akıtılır), **video önizleme**, **sahne görseli galerisi** ve **önceki videolar arşivi** (`output/` içindeki videolar tarih/boyut etiketiyle listelenir; seçilen video arşiv oynatıcısında açılır; liste üretim sonrası ve sayfa açılışında otomatik yenilenir) bulunur.

Edge TTS bağlantıyı açsa bile boş ses döndürürse kalitesiz bir yedeğe geçilmez. Bir sahnenin sesi üretilemezse sessiz video oluşturulmaz; Web UI logunda açık hata gösterilir.

**Dış erişim:**
- Sunucu `0.0.0.0` üzerine bind olduğu için ağdaki diğer cihazlar `http://<sunucu-ip>:7860` adresinden bağlanabilir. `GRADIO_PORT` doluysa uygulama otomatik olarak sonraki boş portu kullanır; `--port` ile açıkça seçilen dolu port ise değiştirilmez ve anlaşılır bir hata verir.
- `GRADIO_SHARE=true` (varsayılan) ile Gradio `https://<id>.gradio.live` adresinde **public link** üretir — internetten herhangi bir yerden açılabilir. Link konsola yazılır ve `output/webui_url.txt` dosyasına kaydedilir. Public linkler geçicidir (Gradio tarafından ~72 saat sonra kapanır).
- **Güvenlik**: Public link açıkken API anahtarları sunucuda kalır (arayüzde gösterilmez), ancak linke erişen herkes video üretimi tetikleyebilir. `.env`'de parola tanımlayın:

```bash
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860
GRADIO_SHARE=true
GRADIO_AUTH_USER=admin
GRADIO_AUTH_PASS=guclu-parola
```

> Not: Web arayüzünde görsel **onay akışı yoktur** (terminaldeki Enter/d/s/q akışı web'e uygun değil); görseller otomatik onaylanır.

### Google Colab CLI ile Uzaktan Çalıştırma

YouTube Shorts pipeline'ını Google Colab'ın bulut tabanlı GPU/TPU kaynaklarında çalıştırın.

#### Colab CLI Kurulumu

```bash
uv tool install google-colab-cli
```

#### Kullanım

```bash
# CPU ile çalıştır
.venv/bin/python generate_shorts.py "Konu başlığı" --colab

# T4 GPU ile çalıştır
.venv/bin/python generate_shorts.py "Konu başlığı" --colab --gpu T4

# A100 GPU ile çalıştır
.venv/bin/python generate_shorts.py "Konu başlığı" --colab --gpu A100

# Özel session adı ve yüksek bellek ile
.venv/bin/python generate_shorts.py "Konu başlığı" --colab --gpu L4 --session my-session --high-mem
```

#### Desteklenen GPU'lar

| GPU | Açıklama |
|-----|----------|
| T4 | Entry-level GPU, ücretsiz/Pro plan |
| L4 | Orta seviye GPU, Pro+ plan |
| G4 | Yüksek performans GPU |
| H100 | Üst seviye GPU |
| A100 | Profesyonel GPU, Pro+ plan |

#### Desteklenen TPU'lar

| TPU | Açıklama |
|-----|----------|
| v5e1 | TPU v5e (1 chiplet) |
| v6e1 | TPU v6e (1 chiplet) |

#### Çevre Değişkenleri ile Yapılandırma

`.env` dosyasına Colab ayarları ekleyebilirsiniz:

```
COLAB_ENABLED=true
COLAB_GPU=T4
COLAB_SESSION_NAME=shorts-gen
COLAB_HIGH_MEM=false
```

#### Nasıl Çalışır

1. `colab new` — Uzaktan Colab VM oluşturulur
2. `colab install` — Bağımlılıklar VM'ye kurulur
3. `colab upload` — Proje dosyaları VM'ye yüklenir
4. `colab exec` — Shorts pipeline'ı VM'de çalıştırılır
5. `colab download` — Üretilen video yerel makineye indirilir
6. `colab stop` — VM serbest bırakılır

## Proje Yapısı

```
appShorts/
├── .env                    # API anahtarları
├── .env.example            # API anahtarı şablonu
├── requirements.txt        # Bağımlılıklar
├── generate_shorts.py      # CLI giriş noktası
├── webui.py                # Gradio web arayüzü (dış erişim + public link)
├── config.py               # Yapılandırma
├── services/
│   ├── scene_generator.py  # Gemini veya Cloudflare CLI ile sahne üretimi
│   ├── cloudflare_cli.py   # cf OAuth oturumu + Workers AI CLI çağrısı
│   ├── image_generator.py  # Cloudflare/Gemini görsel üretimi + onay akışı
│   ├── cloudflare_tts.py   # Cloudflare Workers AI seslendirme
│   ├── voice_generator.py  # Edge TTS seslendirme
│   ├── subtitle_sync.py    # Altyazı zamanlama (kelime sınırları + crossfade zaman çizelgesi)
│   ├── video_assembler.py  # MoviePy ile montaj
│   └── colab_runner.py     # Google Colab CLI entegrasyonu
├── pipeline/
│   └── shorts_pipeline.py  # Ana iş akışı
├── output/                 # Üretilen videolar (shorts_TIMESTAMP.mp4)
└── .tmp/                   # Geçiciler: images/, audio/, video/ (moviepy temp), subtitles.srt
```

## Teknik Detaylar

- **Video Formatı**: 1080x1920 (9:16 dikey), MP4
- **Ses**: Türkçe (tr-TR-AhmetNeural) — **TTS sonrası sahne süresi gerçek ses süresine + 0.4sn senkronlanır** (ses erken kesilmez, sahne geçişi sesi bölmez). **403/throttle backoff**: deneme arası üstel bekleme (`TTS_MAX_RETRIES=5`, `TTS_RETRY_BASE_DELAY=3.0` sn) + sahneler arası `TTS_SCENE_DELAY=1.5` sn
- **Altyazı**: TikTok akordiyon karaoke — siyah zemin, sarı metin, konuşulan kelime beyaz vurgulu. Zamanlama Edge TTS WordBoundary ile **gerçek ses-kelime sınırlarından** (uyumlu); kelimeler disjoint 3'lü gruplara ayrılır (**tekrar yok**); kutu yüksekliği metne göre büyük (taşma yok), alt konum, NotoSans-Bold 52pt. **SRT senkronizasyonu**: satır sınırları gerçek kelime zamanlarından alınır (satır, ilk kelime konuşulduğunda başlar, son kelime bittiğinde biter) ve sahne ofsetleri montajdaki crossfade zaman çizelgesiyle aynı formülle hesaplanır → **altyazı sesle birebir uyumlu**. Kelime sınırı yoksa (ör. ttsai) karakter-oranlı dağıtım kullanılır
- **Çıktı yolları**: biten videolar `output/shorts_TIMESTAMP.mp4`; **tüm geçiciler** (sahne görselleri, ses dosyaları, altyazı, moviepy render temp) proje kökündeki `.tmp/` klasöründe
- **Görsel sağlayıcı zinciri**: Cloudflare Workers AI → (401/403 olursa) Pollinations.ai (anahtarsız) → Gemini → prosedürel gradient. Token geçersizse zincir durmaz, sahneler yine gerçek görselle üretilir
- **AI Model**: Gemini (gemini-3.5-flash-lite)
- **Cloudflare Workers AI**: Varsayılan **`cf` CLI OAuth oturumu** (`--cf-login`) API anahtarı gerektirmez; CLI profili senaryo metni, görsel ve TTS çağrılarında kullanılır. Aktif modeller: `@cf/meta/llama-3.2-3b-instruct` (metin), `@cf/black-forest-labs/flux-1-schnell` (görsel, `steps=4`), `@cf/myshell-ai/melotts` (WAV ses; Türkçe desteklemez). `CLOUDFLARE_ACCOUNT_ID`/`CLOUDFLARE_API_TOKEN` REST yolu geriye dönük alternatiftir. **Boş resim kontrolü**: her görsel çözülüp 32x32 grayscale std ile ölçülür; solid/boş/kirli veri reddedilir ve 1 kez taze seed ile yeniden denenir. Token/OAuth yoksa → Pollinations/Gemini fallback → prosedürel gradient. `IMAGE_PROVIDER=gemini` ile eski zincir: Gemini failover → kota dolunca Cloudflare → Pollinations.ai → gradient. Sahne başına N aday + terminal onay akışı (Enter=onayla, d=değiştir, s=atla, q=iptal; TTY'siz otomatik)
- **Colab CLI**: Uzaktan GPU/TPU desteği (T4, L4, G4, H100, A100)

## API Limitleri

- **Gemini**: Metin modelleri free-tier destekli; görsel model kotası dolabilir
  (429 RESOURCE_EXHAUSTED → akış siyah zeminle devam eder, kota sıfırlandığında görseller üretilir)

## Lisans

MIT
