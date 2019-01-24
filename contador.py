# neste código utiliza-se o opencv e a linguegem python
# para a utilização do YOLOv3
#
import cv2 as cv
import numpy as np


# Inicialização dos parâmetros
confThreshold = 0.4  # limiar de confiança, valores abaixo deste não são contabilizados
nmsThreshold = 0.4  # Non-maximum suppression, remove as marcações (boxes) redundantes
inpWidth = 700  # largura da imagem de entrada submetida a rede
inpHeight = 700  # Altura da imagem de entrada submetida a rede

# Carrega o nome das classes
classesFile = "coco.names"; #todas os objetos treinados pelo modelo
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


# Arquivos com as configurações e pesos da rede
modelConfiguration = "yolov3.cfg";  # configurações da rede
modelWeights = "yolov3.weights";  # configurações de pesos

# Carrega a rede DarkNet utilizando as informações de configurações e pesos
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# Configura o local de execução do código, neste caso a CPU
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


# Retorna o nome das camadas de saída
def getOutputsNames(net):
    # Retorna o nome de todas as camadas na rede
    layersNames = net.getLayerNames()

    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Desenha um caixa delimitadora em volta do objeto detectado
def drawPred(classId, conf, left, top, right, bottom):
    # Desenha a caixa delimitadora
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # pega o nome do rótulo e o valor de confiança e cria uma string
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Apresenta a label no topo da caixa delimitadora
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)



# Remove as caixas delimitadores com baixo valor de confiança usando non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Analisa todas as caixas delimitadoras e mantém apenas aquelas com alto valor
    # de confiança, em seguida, atribuí o rótulo da classe com a classe de maior pontuação
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:] # vetor com os valores de confiança para cada classe contina em coco.
            classId = np.argmax(scores) # armazena o índice da classe de de confiança
            confidence = scores[classId] # armazena o valor de confiança deste objeto

            # verifica se este valor é pequeno, ou seja, se deverá ser considerado
            if classes[classId] == "person" and confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Realiza a operação non maximum suppression para eliminar caixas redundantes
    # sobrepostas com baixa confiança
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], (i+1) , left, top, left + width, top + height)

    # indices contém a quantidade de objetos, no caso da imagem em questão
    # como só há pessoas, então a quantidade de elementos em indice será
    # a predição da quantidade de pessoas na imagem
    print("total de pessoas: ", np.size(indices))


frame = cv.imread("pessoas.jpg")
outputFile = "pessoas" + '_result.jpg'

# O formato blob é o formato de imagem de entrada para a rede neural,
# assim, a função abaixo cria um blob 4D da imagem de entrada.
# Além disso, esta imagem será normalizada com valores de 0 a 1
# e redimensionada para os valores configurados em inpWidth e inpHeight.
blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

# Configura a entrada da rede
net.setInput(blob)

# A saída consiste em uma lista de caixas delimitadoras de vários objetos
outs = net.forward(getOutputsNames(net))
# A etapa seguinte é remover as caixas delimitadoras com valor baixo de confiança.
# Este valor é determinado no início do código pela variável confThreshold
postprocess(frame, outs)

#apresenta a imagem com a quantidade de pessoas
cv.imshow("Quantidade de pessoas na imagem", frame)
cv.imwrite("numero_de_pessoas.png",frame)
cv.waitKey()
