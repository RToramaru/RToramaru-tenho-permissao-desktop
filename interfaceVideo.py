import tkinter as tk
from PIL import Image
from PIL import ImageTk
import arqVideo


def encontraPlaca():
    result = upload.get('1.0', 'end')
    text_widow = canvas.create_window(200, 350, anchor=tk.NW, window=textResult)
    arqVideo.conectar(result)
    textResult.config(text='Reconhecendo...', fg='green', bg='#39b1fd', font=("Courier", 20))

def info():
    newWindow = tk.Toplevel(root)
    newWindow.geometry('{}x{}'.format(200, 200))
    labelInfo = tk.Label(newWindow, text='Sistema Tenho permissão ?\n'
                                            'Desenvolvido por \n'
                                         'Rafael Almeida Soares')
    newWindow.configure(background = '#627efc')
    labelInfo.configure(background='#627efc', fg='white', height=20)
    labelInfo.pack()

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
upload = tk.Text(root, fg='blue')
upload.config(height=1.4, width=27)
button_window = canvas.create_window(200, 200, anchor=tk.NW, window=upload)
detectar = tk.Button(root, text='Detectar', fg='blue', command=encontraPlaca)
detectar.config(height=1, width=30)
button_window = canvas.create_window(200, 230, anchor=tk.NW, window=detectar)
inputImagem = tk.Label(root)
textResult = tk.Label(root)
meuMenu = tk.Menu(root)
meuMenu.add_command(label='Sobre', command=info)
root.config(menu=meuMenu)
root.mainloop()