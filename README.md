<p align="center">
<img width="400" height="400" src="datos/APMLogo.png">
</p>  


# Apple Panic Machine


It does alert you about potentially malicious apples in the vicinity. It cannot (yet) distinguish between lawful apples or threatening ones. 

_Stay tune for future versions._

Detecta manzanas (y a veces móviles) en tiempo real con una Raspberry Pi + Coral Edge TPU. Modelo SSD MobileNet V2 (COCO) en TFLite. Si aparece una manzana, suena la alarma. Empezó como práctica del Máster de Ciencia de Datos (UV, 2022) para probar inferencia en la TPU; lo de panicar es el chiste.

English version of the same thing: object detection on a Pi with a Coral accelerator. Apples set off sound clips, phones get an "okey". Still can't tell a friendly apple from a hostile one.


## Archivos

- `script_for_raspy.py` — detección en vivo con ventana de OpenCV  
- `sin_window_script_for_raspy.py` — lo mismo pero sin ventana (útil en la Pi a pelo)  
- `pruebas_deteccion_objetos_tflite.ipynb` — el notebook de pruebas / apuntes  
- `modelos/` — tflite + labels (y en `own_compilation/` el log del compilador Edge TPU)  
- `datos/` — logo y audios  
- `slides/` — pdfs de las presentaciones  


## Cómo tirarlo

En la Pi, desde la raíz del repo:

```bash
sudo apt install mplayer
pip install -r requirements.txt
# tflite-runtime en Coral/Pi: mejor siguiendo la guía de coral.ai

python script_for_raspy.py
# o sin ventana:
python sin_window_script_for_raspy.py
```

`use_TPU=True` por defecto. Si no tienes Coral, ponlo a `False` y carga el modelo sin `_edgetpu`. Salir con `q` en la ventana.


## Extra

_Presentación sobre la Edge TPU y la implementación de modelos en ella:_    
https://docs.google.com/presentation/d/1p_mAIIVx5xyQ_UkVcDQevtst1Gh_unBkOfQpVFQJQNw/edit?usp=sharing  

_Presentación sobre la Apple Panic Machine:_     
https://docs.google.com/presentation/d/1SQDxS9tib5La0J_XLj54_IejUq0ahExqXJZbw4D4Jhk/edit?usp=sharing  
