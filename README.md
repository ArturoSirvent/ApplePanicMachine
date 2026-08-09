<p align="center">
  <img width="280" alt="Apple Panic Machine logo" src="datos/APMLogo.png">
</p>

# Apple Panic Machine

It alerts you about potentially malicious apples in the vicinity. It cannot (yet) distinguish between lawful apples and threatening ones.

_Stay tuned for future versions._

---

## Qué es esto / What this is

**ES:** Proyecto de detección de objetos en tiempo real sobre Raspberry Pi con Coral Edge TPU. Usa un SSD MobileNet V2 (COCO) en formato TFLite. Si aparece una manzana (o un móvil), suena una alarma. Empezó como práctica del Máster de Ciencia de Datos (UV, 2022) para probar inferencia en Edge TPU; el resto es del chiste.

**EN:** Real-time object detection on a Raspberry Pi with a Coral Edge TPU. Runs an SSD MobileNet V2 (COCO) TFLite model. Spot an apple (or a phone) and it plays an alarm. Started as a Master's project (Data Science, UV, 2022) to try Edge TPU inference; the panic part is the joke.

---

## Hardware

- Raspberry Pi (tested with a camera / webcam)
- [Coral USB Accelerator](https://coral.ai/) (Edge TPU) — optional but intended path (`use_TPU=True`)
- Speakers (audio via `mplayer`)

Without the TPU, set `use_TPU=False` in the scripts and load the non-`edgetpu` model.

---

## Repo layout

```
.
├── script_for_raspy.py              # live detection + OpenCV window
├── sin_window_script_for_raspy.py   # same loop, no display (headless / Pi without screen)
├── pruebas_deteccion_objetos_tflite.ipynb  # notes + experiments (Edge TPU / TFLite)
├── modelos/
│   ├── coco_labels.txt
│   ├── tf2_ssd_mobilenet_v2_coco17_ptq.tflite
│   ├── tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite
│   └── own_compilation/             # Edge TPU compile logs + copies
├── datos/                           # logo + sound clips
└── slides/                          # PDFs from the talks
```

---

## Run

From the repo root, on the Pi:

```bash
# system deps (Debian/Raspbian-ish)
sudo apt install mplayer

# Python deps — on Coral/Pi, prefer the Coral install guide for tflite-runtime
pip install -r requirements.txt
```

With preview window:

```bash
python script_for_raspy.py
```

Without window (prints + audio only):

```bash
python sin_window_script_for_raspy.py
```

Press `q` to quit when the OpenCV window is open.

What it reacts to (defaults in the scripts):

| Detection   | Sound                    |
|-------------|--------------------------|
| `apple`     | `oh_no2_crop.mp3` or `nogod_crop.mp3` (closer → louder freakout) |
| `cell phone`| `okey.mp3`               |

Confidence threshold is ~0.65. Paths assume you run from the repo root (`./modelos`, `./datos`).

---

## Model

Quantized SSD MobileNet V2 trained on COCO, compiled for Edge TPU. Labels in `modelos/coco_labels.txt`. Compile notes under `modelos/own_compilation/`.

Most ops run on the TPU; a few fall back to CPU (see the compile log).

---

## Slides / presentaciones

Local PDFs in `slides/`. Online copies:

- [Edge TPU — how models run on it](https://docs.google.com/presentation/d/1p_mAIIVx5xyQ_UkVcDQevtst1Gh_unBkOfQpVFQJQNw/edit?usp=sharing)
- [Apple Panic Machine](https://docs.google.com/presentation/d/1SQDxS9tib5La0J_XLj54_IejUq0ahExqXJZbw4D4Jhk/edit?usp=sharing)

---

## Notes

Code is from an exploratory notebook session moved to `.py` scripts. Comments are mostly in Spanish. It works as a demo, not as a polished product.

Useful references from when this was written: [Coral docs](https://coral.ai/docs/edgetpu/models-intro/), [EdjeElectronics TFLite on Pi](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi).
