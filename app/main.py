from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Kontur algılama fonksiyonu
def getContours(img, cThr=[100, 100], minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial, kernel, iterations=2)
    contours, _ = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalCountours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
    return img, finalCountours

# Dört köşeyi yeniden sıralayan fonksiyon
def reorder(myPoints):
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# Görüntüyü perspektif olarak yeniden boyutlandırma
def warpImg(img, points, w, h, pad=20):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [h, 0], [0, w], [h, w]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (h, w))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    return imgWarp

# İki nokta arasındaki mesafeyi hesaplama
def findDis(pts1, pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2) ** 0.5

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.form['image']
    # Base64 kodlu görüntü verisini çözme
    image_data = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is not None:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        imgContours, conts = getContours(frame, minArea=1000, filter=4)
        if len(conts) != 0:
            biggest = conts[0][2]
            imgWarp = warpImg(frame, biggest, 297 * 3, 210 * 3)
            imgContours2, conts2 = getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False)
            if len(conts2) != 0:
                for obj in conts2:
                    cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                    nPoints = reorder(obj[2])
                    nW = round((findDis(nPoints[0][0] // 3, nPoints[1][0] // 3)), 1)  # mm cinsine çevirildi
                    nH = round((findDis(nPoints[0][0] // 3, nPoints[2][0] // 3)), 1)  # mm cinsine çevirildi
                    nW2 = round((findDis(nPoints[1][0] // 3, nPoints[3][0] // 3)), 1)  # mm cinsine çevirildi
                    nH2 = round((findDis(nPoints[2][0] // 3, nPoints[3][0] // 3)), 1)  # mm cinsine çevirildi
                    x, y, w, h = obj[3]
                    cv2.putText(imgContours2, 'En : {}mm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                (255, 0, 255), 2)
                    cv2.putText(imgContours2, 'Boy : {}mm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1.5, (255, 0, 255), 2)
                frame = imgContours2
        # İşlenmiş görüntüyü base64 formatında geri gönderme
        _, buffer = cv2.imencode('.jpg', frame)
        response = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': 'data:image/jpeg;base64,' + response})
    else:
        return jsonify({'error': 'Görüntü işlenemedi'}), 400

if __name__ == "__main__":
    app.run(debug=True)
