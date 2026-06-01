
# Practica 1 - Vision Artificial

# Integrantes del grupo:
#   - Joel Domené Álvaro
#   - Nicolas Wenceslao Muñoz
#   - Jorge Bernabé Molinero

import argparse
import cv2
import numpy as np
import os
import csv


# DETECTOR 1: MSER + Correlacion azul
def detector_mser(imagen, mascara_ideal, params=None):
    """
        Detector principal usando MSER. Pasos:
            1. Pasar a grises
            2. MSER detecta regiones
            3. Filtrar por aspect ratio
            4. Expandir bbox
            5. Calcular score con mascara azul
    """
    
    detecciones = []
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # mejorar contraste con CLAHE 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Detector MSER con parametros ajustados: sacado de stackoverflow: https://stackoverflow.com/questions/17647500
    # Parametros posicionales: delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size
    mser = cv2.MSER_create(5, 100, 35000, 0.25, 0.2, 200, 1.01, 0.003, 5)
    
    # Detectar regiones
    regions, bboxes = mser.detectRegions(gray)
    
    for bbox in bboxes:
        x, y, w, h = bbox
        
        # Filtrar por tamaño mínimo, máximo, ratio (anchura >= altura) 
        if w < 35 or h < 25:
            continue
        img_h, img_w = imagen.shape[:2]
        if w > img_w * 0.6 or h > img_h * 0.6:
            continue
        ratio = w / h if h > 0 else 0
        if ratio < 0.4 or ratio > 4.5:
            continue
        
        # Expandir un poco para incluir borde blanco
        ex, ey, ew, eh = expandir_bbox(x, y, w, h, 0.0, imagen.shape)
        
        # Recortar la ventana
        ventana = imagen[ey:ey+eh, ex:ex+ew]
        if ventana.size == 0:
            continue
        
        # Calcular puntuacion
        puntos = calcular_puntos(ventana, mascara_ideal)
        
        # Umbral para aceptar como panel
        if puntos > 0.3:
            x1 = ex
            y1 = ey
            x2 = ex + ew
            y2 = ey + eh
            detecciones.append((x1, y1, x2, y2, round(puntos, 2)))
    
    return detecciones


# DETECTOR 2: Hough Lines + Color
# Detector extra usando lineas de Hough para buscar rectangulos y luego verificar con color azul.
def detector_hough(imagen, mascara_ideal):
    detecciones = []
    
    # primero buscamos regiones azules grandes
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    # Rango de azules
    azul_inferior = np.array([100, 70, 40])
    azul_superior = np.array([135, 255, 255])
    mascara_azul = cv2.inRange(hsv, azul_inferior, azul_superior)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mascara_azul = cv2.morphologyEx(mascara_azul, cv2.MORPH_CLOSE, kernel, iterations=2)
    mascara_azul = cv2.morphologyEx(mascara_azul, cv2.MORPH_OPEN, kernel, iterations=1)
    contornos, _ = cv2.findContours(mascara_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < 800:  # muy pequeño
            continue
        
        # obtener bounding rect
        x, y, w, h = cv2.boundingRect(contorno)
        
        # Filtrar regiones demasiado grandes y ratio
        img_h, img_w = imagen.shape[:2]
        if w > img_w * 0.5 or h > img_h * 0.5:
            continue
        if h == 0:
            continue
        ar = w / h
        if ar < 0.4 or ar > 4.5:
            continue
        
        # Verificar que el rectangulo es azul con la mascara
        ex, ey, ew, eh = expandir_bbox(x, y, w, h, 0.0, imagen.shape)
        ventana = imagen[ey:ey+eh, ex:ex+ew]
        if ventana.size == 0:
            continue
        
        puntos = calcular_puntos(ventana, mascara_ideal)
        
        # Umbral para aceptar como panel
        if puntos > 0.3:
            
            # Verificar bordes de Canny + Hough
            gray_ventana = cv2.cvtColor(ventana, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_ventana, 50, 150)
            lineas = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                      minLineLength=min(ew, eh)//4, maxLineGap=10)
            
            bonus = 0.0
            if lineas is not None and len(lineas) > 2:
                # Contar lineas horizontales y verticales
                lineas_h = 0
                lineas_v = 0
                for linea in lineas:
                    x1l, y1l, x2l, y2l = linea[0]
                    angulo = abs(np.arctan2(y2l-y1l, x2l-x1l) * 180 / np.pi)
                    if angulo < 15 or angulo > 165:
                        lineas_h += 1
                    elif 75 < angulo < 105:
                        lineas_v += 1
                
                if lineas_h >= 1 and lineas_v >= 1:
                    bonus = 0.05  # subimos un poco el score
            
            puntos_final = min(puntos + bonus, 1.0)
            x1 = ex
            y1 = ey
            x2 = ex + ew
            y2 = ey + eh
            detecciones.append((x1, y1, x2, y2, round(puntos_final, 2)))
    
    return detecciones


# DETECTOR COMBINADO
def detector_combinado(imagen, mascara_ideal):
    #Combina MSER y Hough
    dets_mser = detector_mser(imagen, mascara_ideal)
    dets_hough = detector_hough(imagen, mascara_ideal)
    
    # Juntar todas las detecciones
    todas = dets_mser + dets_hough
    
    # Aplicar NMS para quitar duplicados
    resultado = nms(todas, umbral_iou=0.3)
    
    return resultado


# Guardar resultados
def guardar_resultados(detecciones_por_imagen, fichero_salida):
    # Guarda las detecciones en formato resultado.txt
    with open(fichero_salida, 'w') as f:
        for nombre_img, dets in sorted(detecciones_por_imagen.items()):
            for det in dets:
                x1, y1, x2, y2, score = det
                # formato: nombre;x1;y1;x2;y2;tipo;score
                linea = f"{nombre_img};{x1};{y1};{x2};{y2};1;{score:.2f}\n"
                f.write(linea)


def dibujar_detecciones(imagen, detecciones):
    # Dibuja los rectangulos de las detecciones en la imagen
    img_result = imagen.copy()
    for det in detecciones:
        x1, y1, x2, y2, score = det
        # rectangulo rojo
        cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # texto amarillo con el score
        cv2.putText(img_result, f"{score:.2f}", (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return img_result



# MAIN
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--detector', type=str, nargs="?", default="combinado")
    parser.add_argument(
        '--train_path', default="")
    parser.add_argument(
        '--test_path', default="")

    args = parser.parse_args()

    print("Practica 1 - Deteccion de paneles de autopista")
    print(f"Detector: {args.detector}")
    print(f"Train path: {args.train_path}")
    print(f"Test path: {args.test_path}")

    # Entrenamiento ---
    # Miramos las imagenes de train
    mascara_ideal = crear_mascara_azul_ideal(40, 80)
    
    if args.train_path:
        gt_train = cargar_anotaciones(os.path.join(args.train_path, "gt.txt"))
        print(f"Imagenes de train con anotaciones: {len(gt_train)}")
        
        # miramos unas cuantas imagenes de train para ver el rendimiento (usado para ajustar parametros)
        contador = 0
        for nombre in sorted(gt_train.keys())[:5]:
            img_path = os.path.join(args.train_path, nombre)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  No se pudo leer {img_path}")
                continue
            
            if args.detector == "mser":
                dets = detector_mser(img, mascara_ideal)
            elif args.detector == "hough":
                dets = detector_hough(img, mascara_ideal)
            else:
                dets = detector_combinado(img, mascara_ideal)
            
            anotados = len([a for a in gt_train[nombre] if a[4] == 1])
            print(f"{nombre}: detectados={len(dets)}, anotados={anotados}")
            contador += 1

    # Test
    print("\nImagenes de test")
    
    if not args.test_path:
        print("ERROR: No se ha especificado test_path")
        exit(1)
    
    # Directorio de resultados
    resultado_dir = "resultado_imgs"
    if not os.path.exists(resultado_dir):
        os.makedirs(resultado_dir)
    
    # Diccionario para guardar todas las detecciones
    todas_detecciones = {}
    
    # Listar imagenes de test
    imagenes_test = sorted([f for f in os.listdir(args.test_path) if f.endswith('.png')])
    print(f"Total imagenes de test: {len(imagenes_test)}")
    
    for i, nombre_img in enumerate(imagenes_test):
        img_path = os.path.join(args.test_path, nombre_img)
        imagen = cv2.imread(img_path)
        
        if imagen is None:
            print(f"ERROR leyendo {img_path}")
            continue
        
        # Detectar paneles segun el detector elegido
        if args.detector == "mser":
            detecciones = detector_mser(imagen, mascara_ideal)
        elif args.detector == "hough":
            detecciones = detector_hough(imagen, mascara_ideal)
        else:  # combinado (por defecto)
            detecciones = detector_combinado(imagen, mascara_ideal)
        
        todas_detecciones[nombre_img] = detecciones
        
        # Dibujar y guardar imagen resultado
        img_resultado = dibujar_detecciones(imagen, detecciones)
        cv2.imwrite(os.path.join(resultado_dir, nombre_img), img_resultado)
        
        # Progreso
        if (i+1) % 10 == 0 or i == 0:
            print(f"Procesada {i+1}/{len(imagenes_test)}: {nombre_img} -> {len(detecciones)} detecciones")
    
    # Guardar resultado.txt
    fichero_resultado = "resultado.txt"
    guardar_resultados(todas_detecciones, fichero_resultado)
    print(f"\nResultados guardados en: {fichero_resultado}")
    
    # Total detecciones
    total_deten = sum(len(d) for d in todas_detecciones.values())
    imgs_con_deten = sum(1 for d in todas_detecciones.values() if len(d) > 0)
    print(f"Total detecciones: {total_deten}")
    print(f"Imagenes con detecciones: {imgs_con_deten}/{len(imagenes_test)}")
    print(f"Imagenes resultado guardadas en: {resultado_dir}/")

# Funciones auxiliares
def cargar_anotaciones(path_gt):
    # Carga el fichero gt.txt y devuelve lista de anotaciones
    anotaciones = {}
    with open(path_gt, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) < 6:
                continue
            nombre = row[0]
            x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
            tipo = int(row[5])
            if nombre not in anotaciones:
                anotaciones[nombre] = []
            anotaciones[nombre].append((x1, y1, x2, y2, tipo))
    return anotaciones


def crear_mascara_azul_ideal(alto=40, ancho=80):
    # Crea la mascara ideal de un panel azul (casi todo azul con un borde blanco alrededor)
    mascara = np.zeros((alto, ancho), dtype=np.float32)
    margen_x = int(ancho * 0.1)
    margen_y = int(alto * 0.1)
    mascara[margen_y:alto-margen_y, margen_x:ancho-margen_x] = 1.0
    return mascara


def obtener_mascara_azul(imagen_bgr):
    #Dada una imagen BGR, devuelve mascara binaria de pixeles azules saturados. Usamos HSV
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    
    # Rango de azul en HSV (este es el rango que mejor resultados nos ha dado)
    azul_inferior = np.array([100, 80, 40])
    azul_superior = np.array([130, 255, 255])
    
    mascara = cv2.inRange(hsv, azul_inferior, azul_superior)
    # normalizar a 0-1
    mascara = mascara.astype(np.float32) / 255.0
    return mascara


def calcular_puntos(ventana_bgr, mascara_ideal):
    # Calcula la correlacion entre la ventana y la mascara ideal. Redimensiona la ventana al tamaño de la mascara y compara.
    alto, ancho = mascara_ideal.shape[:2]
    ventana = cv2.resize(ventana_bgr, (ancho, alto))
    mascara_ventana = obtener_mascara_azul(ventana)
    
    # Correlación entre el azul detectado y el azul esperado
    numerador = np.sum(mascara_ventana * mascara_ideal)
    
    # Recall
    total_ideal = np.sum(mascara_ideal)
    if total_ideal == 0:
        return 0.0
    recall = numerador / total_ideal
    
    # Precision
    total_ventana = np.sum(mascara_ventana)
    if total_ventana == 0:
        return 0.0
    precision = numerador / total_ventana
    
    # F1-score (media armónica más robusta que media aritmética)
    if (precision + recall) == 0:
        return 0.0
    puntos = 2 * (precision * recall) / (precision + recall)
    
    return float(puntos)


def expandir_bbox(x, y, w, h, factor, img_shape):
    # Cubrir el borde blanco
    dx = int(w * factor)
    dy = int(h * factor)
    
    nx = max(0, x - dx)
    ny = max(0, y - dy)
    nw = min(img_shape[1] - nx, w + 2*dx)
    nh = min(img_shape[0] - ny, h + 2*dy)
    
    return nx, ny, nw, nh


def nms(detecciones, umbral_iou=0.3):
    # Eliminar detecciones repetidas
    if len(detecciones) == 0:
        return []
    
    # Ordenar por puntos de mayor a menor
    detecciones = sorted(detecciones, key=lambda d: d[4], reverse=True)
    
    resultado = []
    usadas = [False] * len(detecciones)
    
    for i in range(len(detecciones)):
        if usadas[i]:
            continue
        resultado.append(detecciones[i])
        usadas[i] = True
        
        for j in range(i+1, len(detecciones)):
            if usadas[j]:
                continue
            
            # Calcular IoU y contencion
            det1, det2 = detecciones[i], detecciones[j]
            x1 = max(det1[0], det2[0])
            y1 = max(det1[1], det2[1])
            x2 = min(det1[2], det2[2])
            y2 = min(det1[3], det2[3])
            
            inter = 0.0
            if x2 > x1 and y2 > y1:
                inter = (x2 - x1) * (y2 - y1)
                
            area1 = (det1[2] - det1[0]) * (det1[3] - det1[1])
            area2 = (det2[2] - det2[0]) * (det2[3] - det2[1])
            union = area1 + area2 - inter
            
            iou = inter / union if union > 0 else 0
            ios = inter / min(area1, area2) if min(area1, area2) > 0 else 0
            
            # Eliminamos si solapan mucho (IoU) o si una esta dentro de otra (IoS > 0.4)
            if iou > umbral_iou or ios > 0.4:
                usadas[j] = True 
    
    return resultado


def calcular_iou(det1, det2):
    # Calcula Intersection sobre Union entre dos detecciones. Formato: (x1, y1, x2, y2, puntos)
    x1 = max(det1[0], det2[0])
    y1 = max(det1[1], det2[1])
    x2 = min(det1[2], det2[2])
    y2 = min(det1[3], det2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    interseccion = (x2 - x1) * (y2 - y1)
    area1 = (det1[2] - det1[0]) * (det1[3] - det1[1])
    area2 = (det2[2] - det2[0]) * (det2[3] - det2[1])
    union = area1 + area2 - interseccion
    
    if union <= 0:
        return 0.0
    
    return interseccion / union

