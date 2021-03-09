import tkinter as tk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import numpy as np
import arqImagem

def encontraImagem():
    path_image = filedialog.askopenfilename(filetypes=[
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])
    if len(path_image) > 0:
        stream = open(u'{link}'.format(link=path_image), "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        frame = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        imagem= imutils.resize(frame, height=400)
        imageToShow = imutils.resize(imagem, width=400)
        imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imageToShow)
        img = ImageTk.PhotoImage(image=im)
        inputImagem.configure(image=img)
        inputImagem.image = img
        imagem_widow = canvas.create_window(8, 150, anchor=tk.NW, window=inputImagem)
        global placa
        placa = path_image
def encontraPlaca():
    text_widow = canvas.create_window(420, 150, anchor=tk.NW, window=textResult)
    result = arqImagem.conectar(placa)
    textResult.config(text=f'Placa: {result}', bg='#39b1fd', fg='white')
placa = None
IMAGE_PATH = 'imagens/fundo.jpg'
WIDTH, HEIGTH = 600, 600
root = tk.Tk()
root.geometry('{}x{}'.format(WIDTH, HEIGTH))
canvas = tk.Canvas(root, width=WIDTH, height=HEIGTH)
canvas.pack()
img = ImageTk.PhotoImage(Image.open(IMAGE_PATH).resize((WIDTH, HEIGTH), Image.ANTIALIAS))
canvas.background = img
bg = canvas.create_image(0, 0, anchor=tk.NW, image=img)
upload = tk.Button(root, text="Upload imagem", fg='blue', command = encontraImagem)
upload.config(height=1, width=20)
button_window = canvas.create_window(420, 70, anchor=tk.NW, window=upload)
detectar = tk.Button(root, text="Detectar", fg='blue', command=encontraPlaca)
detectar.config(height=1, width=20)
button_window = canvas.create_window(420, 100, anchor=tk.NW, window=detectar)
inputImagem = tk.Label(root)
textResult = tk.Label(root)
root.mainloop()