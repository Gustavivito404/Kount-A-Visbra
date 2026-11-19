import cv2
import numpy as np
import os

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

CAMERA_INDEX = 0

ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Distribución:
#
# ID 0 ---------------- ID 1
#  |                      |
#  |      ZONA ÚTIL       |
#  |                      |
# ID 3 ---------------- ID 2

ID_TL = 0
ID_TR = 1
ID_BR = 2
ID_BL = 3

REQUIRED_IDS = [ID_TL, ID_TR, ID_BR, ID_BL]

# ============================================================
# MEDIDAS REALES
# ============================================================

# Tamaño real de un lado del marcador ArUco en milímetros.
# CAMBIA ESTE VALOR POR EL REAL.
ARUCO_SIZE_MM = 40.0

# Dimensiones reales de la zona útil delimitada por
# las esquinas internas de los cuatro ArUco.
#
# CAMBIA ESTOS VALORES POR LOS REALES.
WORKSPACE_WIDTH_MM = 300.0
WORKSPACE_HEIGHT_MM = 300.0

# Resolución de la imagen rectificada.
#
# 3 px/mm:
# 300 mm -> 900 píxeles
#
# Un valor mayor mejora la resolución de las mediciones,
# aunque no crea información que la cámara no tenga.
PIXELS_PER_MM = 3.0

RECTIFIED_WIDTH = int(WORKSPACE_WIDTH_MM * PIXELS_PER_MM)
RECTIFIED_HEIGHT = int(WORKSPACE_HEIGHT_MM * PIXELS_PER_MM)

# ============================================================
# DETECCIÓN DE OBJETOS / SATURACIÓN
# ============================================================

# Diferencia mínima respecto a la mesa vacía.
# Si detecta mucho ruido, aumenta este valor.
# Si no detecta las piezas, disminúyelo.
DIFF_THRESHOLD = 25

# Área mínima para considerar una región como objeto.
# Elimina ruido pequeño.
MIN_OBJECT_AREA_PX = 250

# Área mínima utilizada para saturación.
MIN_COMPONENT_AREA = 100

# Margen que se ignora alrededor de la imagen rectificada.
# Ayuda a evitar falsos contornos en los bordes.
BORDER_MARGIN = 5

BACKGROUND_FILE = "kounta_mesa_vacia.png"

# Si quieres ver la máscara de detección para diagnóstico:
SHOW_DEBUG_MASK = False


# ============================================================
# CREAR DETECTOR ARUCO
# ============================================================

def create_aruco_detector():

    dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICT
    )

    if hasattr(cv2.aruco, "ArucoDetector"):

        parameters = cv2.aruco.DetectorParameters()

        detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters
        )

        return detector

    else:

        parameters = cv2.aruco.DetectorParameters_create()

        return dictionary, parameters


ARUCO_DETECTOR = create_aruco_detector()


# ============================================================
# DETECTAR MARCADORES
# ============================================================

def detect_aruco(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    if hasattr(cv2.aruco, "ArucoDetector"):

        corners, ids, rejected = \
            ARUCO_DETECTOR.detectMarkers(gray)

    else:

        dictionary, parameters = \
            ARUCO_DETECTOR

        corners, ids, rejected = \
            cv2.aruco.detectMarkers(
                gray,
                dictionary,
                parameters=parameters
            )

    markers = {}

    if ids is None:
        return markers, corners, ids

    ids_flat = ids.flatten()

    for marker_corners, marker_id in zip(
        corners,
        ids_flat
    ):

        pts = marker_corners.reshape(
            4, 2
        ).astype(np.float32)

        markers[int(marker_id)] = pts

    return markers, corners, ids


# ============================================================
# MEDICIÓN INDIVIDUAL DE ARUCO
# ============================================================

def euclidean_distance(p1, p2):

    return float(
        np.linalg.norm(
            np.array(p1) - np.array(p2)
        )
    )


def measure_marker(marker_corners):

    p0, p1, p2, p3 = marker_corners

    sides = [
        euclidean_distance(p0, p1),
        euclidean_distance(p1, p2),
        euclidean_distance(p2, p3),
        euclidean_distance(p3, p0)
    ]

    average_px = np.mean(sides)

    if average_px <= 0:
        return 0.0, 0.0

    mm_per_px = ARUCO_SIZE_MM / average_px

    return average_px, mm_per_px


# ============================================================
# OBTENER ZONA ÚTIL
# ============================================================

def get_inner_workspace_points(markers):

    # ArUco 0 -> esquina inferior derecha
    TL = markers[ID_TL][2]

    # ArUco 1 -> esquina inferior izquierda
    TR = markers[ID_TR][3]

    # ArUco 2 -> esquina superior izquierda
    BR = markers[ID_BR][0]

    # ArUco 3 -> esquina superior derecha
    BL = markers[ID_BL][1]

    return np.array(
        [TL, TR, BR, BL],
        dtype=np.float32
    )


# ============================================================
# HOMOGRAFÍA
# ============================================================

def calculate_homography(image_points):

    # Coordenadas físicas de la zona útil en mm
    physical_points_mm = np.array([
        [0.0, 0.0],
        [WORKSPACE_WIDTH_MM, 0.0],
        [WORKSPACE_WIDTH_MM, WORKSPACE_HEIGHT_MM],
        [0.0, WORKSPACE_HEIGHT_MM]
    ], dtype=np.float32)

    # Coordenadas correspondientes de la imagen rectificada
    rectified_points = np.array([
        [0.0, 0.0],
        [RECTIFIED_WIDTH - 1, 0.0],
        [RECTIFIED_WIDTH - 1, RECTIFIED_HEIGHT - 1],
        [0.0, RECTIFIED_HEIGHT - 1]
    ], dtype=np.float32)

    H_mm = cv2.getPerspectiveTransform(
        image_points,
        physical_points_mm
    )

    H_rect = cv2.getPerspectiveTransform(
        image_points,
        rectified_points
    )

    return H_mm, H_rect


def pixel_to_mm(point, H_mm):

    p = np.array([
        [[float(point[0]), float(point[1])]]
    ], dtype=np.float32)

    result = cv2.perspectiveTransform(
        p,
        H_mm
    )

    x_mm, y_mm = result[0][0]

    return float(x_mm), float(y_mm)


# ============================================================
# LIMPIAR MÁSCARA
# ============================================================

def remove_small_components(mask):

    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

    clean = np.zeros_like(mask)

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_COMPONENT_AREA:
            clean[labels == label] = 255

    return clean


# ============================================================
# GENERAR MÁSCARA DE OBJETOS
# ============================================================

def generate_object_mask(rectified, background):

    current_gray = cv2.cvtColor(
        rectified,
        cv2.COLOR_BGR2GRAY
    )

    background_gray = cv2.cvtColor(
        background,
        cv2.COLOR_BGR2GRAY
    )

    # Suavizado
    current_gray = cv2.GaussianBlur(
        current_gray,
        (5, 5),
        0
    )

    background_gray = cv2.GaussianBlur(
        background_gray,
        (5, 5),
        0
    )

    # Diferencia respecto a mesa vacía
    difference = cv2.absdiff(
        current_gray,
        background_gray
    )

    # Umbral
    _, mask = cv2.threshold(
        difference,
        DIFF_THRESHOLD,
        255,
        cv2.THRESH_BINARY
    )

    # Operaciones morfológicas
    kernel_open = np.ones(
        (3, 3),
        np.uint8
    )

    kernel_close = np.ones(
        (7, 7),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    mask = remove_small_components(mask)

    # Eliminar bordes
    mask[:BORDER_MARGIN, :] = 0
    mask[-BORDER_MARGIN:, :] = 0
    mask[:, :BORDER_MARGIN] = 0
    mask[:, -BORDER_MARGIN:] = 0

    return mask, difference


# ============================================================
# SATURACIÓN
# ============================================================

def calculate_saturation(mask):

    occupied_pixels = cv2.countNonZero(mask)

    total_pixels = (
        mask.shape[0]
        *
        mask.shape[1]
    )

    if total_pixels == 0:
        return 0.0

    saturation = (
        occupied_pixels / total_pixels
    ) * 100.0

    return saturation


# ============================================================
# DETECTAR Y MEDIR OBJETOS
# ============================================================

def detect_and_measure_objects(
    rectified,
    mask
):

    output = rectified.copy()

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_objects = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_OBJECT_AREA_PX:
            continue

        # Rectángulo rotado mínimo
        rect = cv2.minAreaRect(contour)

        center, size, angle = rect

        width_px, height_px = size

        if width_px <= 0 or height_px <= 0:
            continue

        # Tomar lado largo y lado corto
        long_px = max(
            width_px,
            height_px
        )

        short_px = min(
            width_px,
            height_px
        )

        # Conversión píxel -> mm
        long_mm = (
            long_px / PIXELS_PER_MM
        )

        short_mm = (
            short_px / PIXELS_PER_MM
        )

        area_mm2 = (
            area /
            (PIXELS_PER_MM ** 2)
        )

        box = cv2.boxPoints(rect)

        box = np.int32(box)

        valid_objects.append({
            "contour": contour,
            "box": box,
            "center": center,
            "long_mm": long_mm,
            "short_mm": short_mm,
            "area_mm2": area_mm2,
            "area_px": area
        })

    # Ordenar objetos de izquierda a derecha
    valid_objects.sort(
        key=lambda obj: obj["center"][0]
    )

    for index, obj in enumerate(
        valid_objects,
        start=1
    ):

        box = obj["box"]

        cx = int(obj["center"][0])
        cy = int(obj["center"][1])

        long_mm = obj["long_mm"]
        short_mm = obj["short_mm"]

        # Rectángulo rotado
        cv2.drawContours(
            output,
            [box],
            0,
            (0, 255, 0),
            2
        )

        # Centro
        cv2.circle(
            output,
            (cx, cy),
            4,
            (0, 0, 255),
            -1
        )

        # Texto principal
        text1 = (
            f"Objeto {index}"
        )

        text2 = (
            f"L: {long_mm:.1f} mm"
        )

        text3 = (
            f"A: {short_mm:.1f} mm"
        )

        text_x = max(
            cx - 60,
            5
        )

        text_y = max(
            cy - 35,
            50
        )

        # Fondo para facilitar lectura
        cv2.rectangle(
            output,
            (text_x - 4, text_y - 22),
            (text_x + 145, text_y + 45),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            output,
            text1,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            text2,
            (text_x, text_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            text3,
            (text_x, text_y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    return output, valid_objects


# ============================================================
# DIBUJAR INFORMACIÓN DE LOS ARUCO
# ============================================================

def draw_marker_information(
    frame,
    markers
):

    y_text = 25

    for marker_id in REQUIRED_IDS:

        if marker_id not in markers:
            continue

        corners = markers[marker_id]

        avg_px, mm_per_px = \
            measure_marker(corners)

        cv2.putText(
            frame,
            (
                f"ID {marker_id}: "
                f"{avg_px:.1f}px  "
                f"{mm_per_px:.4f}mm/px"
            ),
            (10, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y_text += 22


# ============================================================
# DIBUJAR ZONA ÚTIL
# ============================================================

def draw_workspace(
    frame,
    workspace_points
):

    pts = workspace_points.astype(
        np.int32
    )

    cv2.polylines(
        frame,
        [pts],
        True,
        (0, 255, 0),
        2
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print(
            "ERROR: No se pudo abrir la camara."
        )

        return

    # Resolución solicitada
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1920
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        1080
    )

    background = None

    # Intentar cargar mesa vacía anterior
    if os.path.exists(BACKGROUND_FILE):

        background = cv2.imread(
            BACKGROUND_FILE
        )

        if background is not None:

            print(
                "Referencia de mesa vacia cargada."
            )

    print()
    print("======================================")
    print(" KOUNT-A")
    print(" ArUco + Homografia + Medicion")
    print(" Saturacion + Deteccion de objetos")
    print("======================================")
    print()
    print("R = guardar nueva referencia vacia")
    print("Q = salir")
    print()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        display = frame.copy()

        markers, corners, ids = \
            detect_aruco(frame)

        # Dibujar ArUco
        if ids is not None:

            cv2.aruco.drawDetectedMarkers(
                display,
                corners,
                ids
            )

        draw_marker_information(
            display,
            markers
        )

        all_detected = all(
            marker_id in markers
            for marker_id in REQUIRED_IDS
        )

        rectified = None
        H_mm = None
        H_rect = None

        # ====================================================
        # LOS 4 ARUCO ESTÁN DISPONIBLES
        # ====================================================

        if all_detected:

            workspace_points = \
                get_inner_workspace_points(
                    markers
                )

            draw_workspace(
                display,
                workspace_points
            )

            H_mm, H_rect = \
                calculate_homography(
                    workspace_points
                )

            rectified = cv2.warpPerspective(
                frame,
                H_rect,
                (
                    RECTIFIED_WIDTH,
                    RECTIFIED_HEIGHT
                )
            )

            cv2.putText(
                display,
                "ArUco: OK",
                (
                    10,
                    display.shape[0] - 50
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ================================================
            # REFERENCIA EXISTENTE
            # ================================================

            if background is not None:

                if (
                    background.shape[:2]
                    !=
                    rectified.shape[:2]
                ):

                    background = cv2.resize(
                        background,
                        (
                            RECTIFIED_WIDTH,
                            RECTIFIED_HEIGHT
                        )
                    )

                mask, difference = \
                    generate_object_mask(
                        rectified,
                        background
                    )

                saturation = \
                    calculate_saturation(mask)

                measured_image, objects = \
                    detect_and_measure_objects(
                        rectified,
                        mask
                    )

                # --------------------------------------------
                # INFORMACIÓN GENERAL
                # --------------------------------------------

                cv2.putText(
                    measured_image,
                    (
                        f"Objetos detectados: "
                        f"{len(objects)}"
                    ),
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    measured_image,
                    (
                        f"Saturacion: "
                        f"{saturation:.2f}%"
                    ),
                    (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    display,
                    (
                        f"Saturacion: "
                        f"{saturation:.2f}%"
                    ),
                    (
                        10,
                        display.shape[0] - 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # Mostrar zona útil con mediciones
                cv2.imshow(
                    "Zona util - Medicion",
                    measured_image
                )

                # Ventana opcional para depurar
                if SHOW_DEBUG_MASK:

                    cv2.imshow(
                        "Mascara de objetos",
                        mask
                    )

            # ================================================
            # NO EXISTE REFERENCIA VACÍA
            # ================================================

            else:

                cv2.putText(
                    rectified,
                    "MESA VACIA: presiona R",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                cv2.imshow(
                    "Zona util - Medicion",
                    rectified
                )

                cv2.putText(
                    display,
                    "Presiona R con la mesa vacia",
                    (
                        10,
                        display.shape[0] - 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        # ====================================================
        # FALTAN ARUCO
        # ====================================================

        else:

            missing = [
                marker_id
                for marker_id in REQUIRED_IDS
                if marker_id not in markers
            ]

            cv2.putText(
                display,
                f"Faltan ArUco: {missing}",
                (
                    10,
                    display.shape[0] - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Ventana principal
        cv2.imshow(
            "Kount-A - Camara",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        # ====================================================
        # GUARDAR REFERENCIA DE MESA VACÍA
        # ====================================================

        if key == ord("r"):

            if all_detected and rectified is not None:

                background = rectified.copy()

                cv2.imwrite(
                    BACKGROUND_FILE,
                    background
                )

                print()
                print(
                    "Referencia de mesa vacia guardada."
                )
                print(
                    BACKGROUND_FILE
                )
                print()

            else:

                print(
                    "No se puede guardar:"
                    " faltan los 4 ArUco."
                )

        # ====================================================
        # SALIR
        # ====================================================

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()