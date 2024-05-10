# BeTech-Hackathon-2024
## Problema 1. CNN
### Enunciado
Después de transcurrir varias horas desde que el barco reportó su última posición, se inicia la búsqueda de los supervivientes. En primer lugar, se establece un equipo de búsqueda conjunto entre varios países, desplegando aviones de sus fuerzas aéreas y guardacostas. Estos aviones despegan y fotografían un área de x kilómetros cuadrados cerca de la última posición conocida del navío. Una vez regresan a la base, comienzan a analizar los datos recopilados.
Los participantes deben desarrollar un sistema de reconocimiento de imagen que analice todas las imágenes capturadas durante la búsqueda. Este sistema debe asignar a cada imagen una clasificación según la probabilidad, en un rango del 0 al 100 o de 0.00 a 1.00 (dependiendo de la herramienta que se utilice), de contener la clase de objeto deseada (barco) y enmarcar el objeto detectado. Se busca implementar el método de detección de objetos.
Se recomienda a los participantes utilizar distintas herramientas de visión computacional como YOLO (You Only Look Once) y TensorFlow para ver cuál de ellas es más precisa.

*0.5 puntos extra si se implementa segmentación de objetos y se detectan otras clases que no sean barcos.

Nota: se valorará la originalidad de la implementación y la limpieza del código

### Dataset:
4000 imágenes .png 80x80 RGB etiquetadas como 'barco' o 'no barco'. Cada imagen tiene como nombre de archivo {etiqueta}_{id escena}_{longitud}_{latitud}.png donde etiqueta = 1 representa 'barco', etiqueta = 0 representa 'no barco'.
El dataset también es proporcionado en un JSON, llamado shipnet.json, que contiene los valores rgb de las imágenes, junto con las etiquetas 0 y 1 de cada imagen.

### Notas:
Durante el transcurso de la Hackathon no fuimos capaces de utilizar modelos de reconocimiento ya entrenados. La solución a la que llegamos fue entrenar nuestro propio modelo de reconocimiento de imágenes.
