import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def dedo_arriba(hand, tip, pip):
    return hand.landmark[tip].y < hand.landmark[pip].y

def dedos_levantados(hand):
    dedos = []
    if dedo_arriba(hand, 4, 3): dedos.append("Pulgar")
    if dedo_arriba(hand, 8, 6): dedos.append("Índice")
    if dedo_arriba(hand, 12, 10): dedos.append("Medio")
    if dedo_arriba(hand, 16, 14): dedos.append("Anular")
    if dedo_arriba(hand, 20, 18): dedos.append("Meñique")
    return dedos

def mano_cerrada(hand):
    tips = [hand.landmark[4], hand.landmark[8], hand.landmark[12], hand.landmark[16], hand.landmark[20]]
    base = [hand.landmark[0]]*5
    d = [distancia(tips[i], base[i]) for i in range(5)]
    return all([x < 0.08 for x in d])

def gesto_mano(hand):
    if not hand: 
        return ""
    if mano_cerrada(hand): 
        return "MANO CERRADA"

    pulgar = dedo_arriba(hand, 4, 3)
    indice = dedo_arriba(hand, 8, 6)
    medio = dedo_arriba(hand, 12, 10)
    anular = dedo_arriba(hand, 16, 14)
    meñique = dedo_arriba(hand, 20, 18)

    if pulgar and not indice and not medio and not anular and not meñique:
        return "LIKE"
    pulgar_tip = hand.landmark[4].y
    pulgar_base = hand.landmark[3].y
    if pulgar_tip > pulgar_base and not indice and not medio and not anular and not meñique:
        return "DISLIKE"
    if indice and medio and not pulgar and not anular and not meñique:
        return "AMOR Y PAZ"
    if distancia(hand.landmark[4], hand.landmark[8]) < 0.05 and medio and anular and meñique:
        return "OKEY"
    return ""

def gesto_ambas_manos(hand_left, hand_right):
    if hand_left and hand_right:
        left_tip = hand_left.landmark[8]
        right_tip = hand_right.landmark[8]
        left_thumb = hand_left.landmark[4]
        right_thumb = hand_right.landmark[4]
        if distancia(left_tip, right_tip) < 0.08 and distancia(left_thumb, right_thumb) < 0.08:
            return "CORAZÓN"
        if distancia(left_thumb, right_thumb) < 0.08:
            return "AVE"
    return ""

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand_left = None
    hand_right = None
    dedos_izq, dedos_der = [], []
    gesto_izq, gesto_der, gesto_doble = "", "", ""
    conteo_izq, conteo_der, conteo_total = 0, 0, 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if handedness.classification[0].label == "Left":
                hand_left = hand_landmarks
                dedos_izq = dedos_levantados(hand_landmarks)
                conteo_izq = len(dedos_izq)
                gesto_izq = gesto_mano(hand_landmarks)
            else:
                hand_right = hand_landmarks
                dedos_der = dedos_levantados(hand_landmarks)
                conteo_der = len(dedos_der)
                gesto_der = gesto_mano(hand_landmarks)

            for idx, landmark in enumerate(hand_landmarks.landmark):
                px, py = int(landmark.x*w), int(landmark.y*h)
                cv2.putText(image, str(idx), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3, cv2.LINE_AA)

    conteo_total = conteo_izq + conteo_der
    gesto_doble = gesto_ambas_manos(hand_left, hand_right)

    if gesto_doble:
        cv2.putText(image, gesto_doble, (w//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (128,0,128), 3, cv2.LINE_AA)
    else:
        if gesto_izq:
            cv2.putText(image, gesto_izq, (w//4 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,255), 3, cv2.LINE_AA)
        if gesto_der:
            cv2.putText(image, gesto_der, (3*w//4 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,0,0), 3, cv2.LINE_AA)

    for i, dedo in enumerate(dedos_izq):
        cv2.putText(image, dedo, (10, 30+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
    for i, dedo in enumerate(dedos_der):
        cv2.putText(image, dedo, (w-200, 30+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2, cv2.LINE_AA)

    cv2.putText(image, f"Derecha: {conteo_der}", (w-200, h-60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 3, cv2.LINE_AA)
    cv2.putText(image, f"Izquierda: {conteo_izq}", (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3, cv2.LINE_AA)
    cv2.putText(image, f"Total: {conteo_total}", (w//2 - 80, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,177,177), 3, cv2.LINE_AA)

    cv2.imshow("Reconocimiento de Gestos", cv2.resize(image, (1280,720)))
    if cv2.waitKey(5) & 0xFF==27: break

cap.release()
cv2.destroyAllWindows()
