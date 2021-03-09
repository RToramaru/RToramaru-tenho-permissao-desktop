from tkinter import *
from tkinter import filedialog
import PIL
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import numpy as np

def elegir_imagen():
    # Especificar los tipos de archivos, para elegir solo a las imágenes
    path_image = filedialog.askopenfilename(filetypes = [
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])

    if len(path_image) > 0:
        global image

        # Leer la imagen de entrada y la redimensionamos
        stream = open(u'{link}'.format(link=path_image), "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)

        frame = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        image= imutils.resize(frame, height=380)

        # Para visualizar la imagen de entrada en la GUI
        imageToShow= imutils.resize(image, width=180)
        imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imageToShow )
        img = ImageTk.PhotoImage(image=im)

        lblInputImage.configure(image=img)
        lblInputImage.image = img

        # Label IMAGEN DE ENTRADA
        lblInfo1 = Label(root, text="IMAGEN DE ENTRADA:")
        lblInfo1.grid(column=0, row=1, padx=5, pady=5)

        # Al momento que leemos la imagen de entrada, vaciamos
        # la iamgen de salida y se limpia la selección de los
        # radiobutton
        selected.set(0)



# Creamos la ventana principal
root = Tk()
# Label donde se presentará la imagen de entrada
lblInputImage = Label(root)
lblInputImage.grid(column=0, row=2)


# Creamos los radio buttons y la ubicación que estos ocuparán
selected = IntVar()
# Creamos el botón para elegir la imagen de entrada
btn = Button(root, text="Elegir imagen", width=25, command=elegir_imagen)
btn.grid(column=0, row=0, padx=5, pady=5)

root.mainloop()
