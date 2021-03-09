import cv2
import numpy as np
import MySQLdb
from datetime import datetime, timedelta
import base64
import math


def conectar(url):

    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    listaCaracteres = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1',
                       '2', '3', '4', '5', '6', '7', '8', '9']

    netVeiculo = cv2.dnn.readNet("pesos/veiculo.weights", "configuracoes/veiculo.cfg")
    netVeiculo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    netVeiculo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    modelVeiculo = cv2.dnn_DetectionModel(netVeiculo)
    modelVeiculo.setInputParams(size=(416, 416), scale=1 / 255)

    netPlaca = cv2.dnn.readNet("pesos/placa.weights", "configuracoes/placa.cfg")
    netPlaca.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    netPlaca.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    modelPlaca = cv2.dnn_DetectionModel(netPlaca)
    modelPlaca.setInputParams(size=(416, 416), scale=1 / 255)

    netRegiao = cv2.dnn.readNet("pesos/regiao.weights", "configuracoes/regiao.cfg")
    netRegiao.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    netRegiao.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    modelRegiao = cv2.dnn_DetectionModel(netRegiao)
    modelRegiao.setInputParams(size=(416, 416), scale=1 / 255)

    netCaractere = cv2.dnn.readNet("pesos/caractere.weights", "configuracoes/caractere.cfg")
    netCaractere.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    netCaractere.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    modelCaractere = cv2.dnn_DetectionModel(netCaractere)
    modelCaractere.setInputParams(size=(416, 416), scale=1 / 255)

    db = MySQLdb.connect(host='localhost', user='root', passwd='', db='registro')
    cur = db.cursor()

    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        classesVeiculo, scoresVeiculo, boxesVeiculo = modelVeiculo.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for c in boxesVeiculo:
            xVeiculo, yVeiculo, wVeiculo, hVeiculo = c
            veiculo = frame[yVeiculo:yVeiculo + hVeiculo, xVeiculo:xVeiculo + wVeiculo]
            classesPlaca, scoresPlaca, boxesPlaca = modelPlaca.detect(veiculo, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            _, im_arr = cv2.imencode('.jpg', veiculo)
            im_bytes = im_arr.tobytes()
            im_b64 = base64.b64encode(im_bytes)
            for d in boxesPlaca:
                xPlaca, yPlaca, wPlaca, hPlaca = d
                placa = veiculo[yPlaca:yPlaca + hPlaca, xPlaca:xPlaca + wPlaca]
                height, width = placa.shape[:2]
                x = 318 / height
                placa = cv2.resize(placa, (0, 0), fx=x, fy=x, interpolation=cv2.INTER_AREA)
                height, width, _ = np.shape(placa)
                avg_color_per_row = np.average(placa, axis=0)
                avg_colors = np.average(avg_color_per_row, axis=0)
                int_averages = np.array(avg_colors, dtype=np.uint8)
                b = int_averages[0]
                g = int_averages[1]
                r = int_averages[2]
                if 0 < b < 85 and 0 < g < 85 and 130 < r < 255:
                    placa = placa - 255
                desfoque = cv2.pyrMeanShiftFiltering(placa, 10, 40)
                cinza = cv2.cvtColor(desfoque, cv2.COLOR_BGR2GRAY)
                cinza = cv2.medianBlur(cinza, 11)
                cinza = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
                classesRegiao, scoresRegiao, boxesRegiao = modelRegiao.detect(cinza, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

                contornos = []
                for e in boxesRegiao:
                    contornos.append(e)
                contornos = sorted(contornos, key=lambda cont: cont[0])
                if int(classesVeiculo[0]) == 1:
                    max_width = max(contornos, key=lambda r: r[0] + r[2])[0]
                    max_height = max(contornos, key=lambda r: r[3])[3]
                    nearest = max_height * 1.4
                    contornos.sort(key=lambda r: (int(nearest * round(float(r[1]) / nearest)) * max_width + r[0]))
                ponto1 = contornos[0]
                ponto2 = contornos[1]
                y1 = ponto1[1]
                y2 = ponto2[1]
                x1 = ponto1[0]
                x2 = ponto2[0]
                m = (y2 - y1) / (x2 - x1)
                m = math.degrees(m)

                (h, w) = desfoque.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, m, 1.0)
                rotated = cv2.warpAffine(cinza, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                classesRegiao, scoresRegiao, boxesRegiao = modelRegiao.detect(rotated, CONFIDENCE_THRESHOLD,
                                                                              NMS_THRESHOLD)
                contornos = []
                for e in boxesRegiao:
                    contornos.append(e)
                contornos = sorted(contornos, key=lambda cont: cont[0])
                if int(classesVeiculo[0]) == 1:
                    max_width = max(contornos, key=lambda r: r[0] + r[2])[0]
                    max_height = max(contornos, key=lambda r: r[3])[3]
                    nearest = max_height * 1.4
                    contornos.sort(key=lambda r: (int(nearest * round(float(r[1]) / nearest)) * max_width + r[0]))
                ocrPlaca = ''
                for f in contornos:
                    xRegiao, yRegiao, wRegiao, hRegiao = f
                    caracte = placa[yRegiao:yRegiao + hRegiao, xRegiao:xRegiao + wRegiao]
                    cinza = cv2.cvtColor(caracte, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    invert = 255 - thresh
                    invert = cv2.cvtColor(invert, cv2.COLOR_GRAY2BGR)
                    altura, largura, channels = caracte.shape
                    imagem = np.zeros((altura * 2, largura * 2, 3), dtype=np.uint8)
                    cv2.rectangle(imagem, (0, 0), (largura * 2, altura * 2), (255, 255, 255), -1)
                    x_offset = int(largura * 0.25)
                    y_offset = int(altura * 0.25)
                    imagem[y_offset:y_offset + invert.shape[0], x_offset:x_offset + invert.shape[1]] = invert

                    classesCaractere, scoresCaractere, boxesCaractere = modelCaractere.detect(imagem, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
                    ocrPlaca += listaCaracteres[int(classesCaractere[0])]

                parte1 = ocrPlaca[:3]
                parte2 = ocrPlaca[3]
                parte3 = ocrPlaca[4]
                parte4 = ocrPlaca[5:]

                parte1 = parte1.replace('1', 'I')
                parte1 = parte1.replace('0', 'O')

                parte2 = parte2.replace('I', '1')
                parte2 = parte2.replace('O', '0')

                parte3 = parte3.replace('I', '1')
                parte3 = parte3.replace('O', '0')

                parte4 = parte4.replace('I', '1')
                parte4 = parte4.replace('O', '0')
                ocrPlaca = parte1 + parte2 + parte3 + parte4
                atual = datetime.now() - timedelta(minutes=5)
                formatted_atual = atual.strftime('%Y-%m-%d %H:%M:%S')
                sql = "SELECT placa FROM dados WHERE data > '{}'".format(formatted_atual)
                cur.execute(sql)
                dataBanco = datetime.now()
                formatted_data = dataBanco.strftime('%Y-%m-%d %H:%M:%S')
                itens = []
                for item in cur.fetchall():
                    itens.append(item[0])
                if ocrPlaca not in itens:
                    parte3 = parte3.replace('1', 'I')
                    parte3 = parte3.replace('0', 'O')
                    ocrPlaca = parte1 + parte2 + parte3 + parte4
                    sql = "INSERT INTO dados (placa, data, imagem) VALUES (%s, %s, %s)"
                    val = (ocrPlaca, formatted_atual, im_b64)
                    cur.execute(sql, val)
                    db.commit()
            db.close()
