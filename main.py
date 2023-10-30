import cv2
import os
import numpy as np
from pymongo import MongoClient

  # criando uma instância do cliente MongoDB
client = MongoClient()

  # Conectando ao servidor MongoDB
db = client.faces

db.faces.insert_one({"chave": "valor"})

'''
classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def gera_id_unico():
    import time
    return int(time.time())

def verifica_usuario_cadastrado():
    return os.path.exists('classificadorLBPH_V1.yml')

def captura_e_treinamento(id):
    amostra = 1
    numeroAmostras = 10

    WebCamera = cv2.VideoCapture(0)
    largura, altura = 640, 480  # Redimensione a imagem para melhor desempenho

    lbph = cv2.face_LBPHFaceRecognizer.create()

    while True:
        conectado, imagem = WebCamera.read()
        imagem = cv2.resize(imagem, (largura, altura))  # Redimensione a imagem
        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        faceDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.1, minSize=(150, 150))

        for (x, y, l, a) in faceDetectadas:
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (220, 220))
            cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
            print("[Foto " + str(amostra) + " capturada com sucesso]")
            amostra += 1

        cv2.imshow("Face", imagem)
        if cv2.waitKey(1) & 0xFF == ord('q') or (amostra >= numeroAmostras + 1):
            break

    print("Faces capturadas com sucesso")
    WebCamera.release()

    faces, ids = getImagemComId()
    faces = [f.tolist() for f in faces]  # Converta 'faces' em uma lista Python
    ids = [int(id) for id in ids]  # Converta 'ids' em uma lista de inteiros
    lbph = cv2.face_LBPHFaceRecognizer_create()
    lbph.train(faces, np.array(ids))  # Certifique-se de passar os rótulos como um numpy array
    lbph.write('classificadorLBPH_V1.yml')
    print("Treinamento concluído ...")

    return lbph


def reconhece_rosto():
    camera = cv2.VideoCapture(0)
    largura, altura = 640, 480  # Redimensione a imagem para melhor desempenho

    lbph = cv2.face_LBPHFaceRecognizer.create()
    lbph.read('classificadorLBPH_V1.yml')

    while True:
        conectado, imagem = camera.read()
        imagem = cv2.resize(imagem, (largura, altura))  # Redimensione a imagem
        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        faceDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.1, minSize=(150, 150))

        for (x, y, l, a) in faceDetectadas:
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (100, 100))
            id, confianca = lbph.predict(imagemFace)

            if confianca < 70:  # Ajuste o valor de confiança conforme necessário
                nome = f"Customer {id}"
            else:

                nome = "Realizando cadastro..."
                lbph = captura_e_treinamento(10)
                print("Novo cadastro realizado")

            cv2.putText(imagem, nome, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        cv2.imshow("Face", imagem)
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    for caminhosImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhosImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhosImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)
    return faces, ids  # Retorna 'faces' e 'ids' diretamente

def main():
    if not verifica_usuario_cadastrado():
        id = gera_id_unico()
        captura_e_treinamento(id)
    reconhece_rosto()

if __name__ == '__main__':
    main()
'''