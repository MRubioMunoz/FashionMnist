#IMPORTACIONES NECESARIAS
#PIL para el tratamiento de imagenes
from PIL import Image,ImageOps
#Importaciones propias de Django y un diccionario donde almaceno los resultados
from django.shortcuts import  render
from .soluciones import soluciones_dic
from PIL import Image
#Librerias necesarias para poder cargar la red neuronal
import numpy as np
from keras.models import load_model 
from tensorflow import keras

# Instanciamos la red antes de cargar la vista para que no este ejecutandose este proceso cada vez que se renderice la vista

modelo = load_model("model.h5")

# Codigo de la vista
# Esta funcion renderiza la vista donde se muestra el formulario de salida y en caso de que se haya subido una imagen (if  request.method == "POST":)
# con la que realizar una predicción, volvera a renderizarla mostrando el resultado. 

def formulario(request):
    
    if  request.method == "POST":

        # Recibimos la imagen que queremos predecir y la almacenamos en formato png para no perder calidad de esta

        fichero = request.FILES['upload']
        imagen_preproceso = Image.open(fichero)
        imagen_preproceso.save('main/imagen.png', "png")

        # Abrimos la imagen previamente almacenada y realizamos una serie de transformaciones en esta
        # Primero se pasa a una escala de grises, luego invertimos los colores y por último la redimensinamos a 28X28 pixeles

        imagen = Image.open("main/imagen.png")
        imagen = imagen.convert('L')
        imagen = ImageOps.invert(imagen)
        imagen = imagen.resize((28,28))
        imagen.save('main/imagenFinal.png', "png")
        
        # Tranformamos la imagen en un array de numpy para tratarla con la red neural

        imagen_a_predecir = np.expand_dims(imagen,0)
        
        #Realizamos la predicción dividiendo el array entre 255 para escalar los valores entre 0 y 1

        predicion = modelo.predict(imagen_a_predecir/255)
        valor = np.argmax(predicion)

        # Renderizamos la misma vista ya con la solución

        return render(request,'main/formulario.html', {"solucion":soluciones_dic.get(valor)})

    # En caso de que sea la primera vez que se renderiza la vista, esta solo mostrará el formulario de subida
    
    return render(request, 'main/formulario.html')