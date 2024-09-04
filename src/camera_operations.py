import numpy as np
import cv2

def calculate_optical_flow(prev_frame, curr_frame):
    """
    Calcula o Optical Flow entre dois frames usando o método Farneback
    e retorna a imagem codificada visualmente.
    """
    # Converter os frames para escala de cinza
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calcular o optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Converter o fluxo em magnitude e ângulo
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Codificar o ângulo como matiz (H), e a magnitude como valor (V)
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255  # Saturação máxima

    # Normalizar o ângulo para estar entre 0 e 180 (OpenCV usa escala de matiz 0-180)
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Normalizar a magnitude para o intervalo 0-255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Converter de HSV para BGR para exibição
    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow, flow_img  # Retorna o fluxo e a imagem codificada


def save_optical_flow_image(flow_img, output_path):
    """
    Salva a imagem do Optical Flow no caminho fornecido.
    """
    cv2.imwrite(output_path, flow_img)


def calculate_background_motion(flow, mask=None):
    """
    Calcula o movimento médio do background.
    """
    if mask is not None:
        flow = flow[mask == 0]  # Ignorar áreas com competidores

    avg_motion = np.mean(flow, axis=0)  # Movimento médio do fundo
    return avg_motion


def estimate_angle_correction_factor(centroids, frame_width, max_correction=1.0):
    """
    Estima o fator de correção de ângulo com base na distribuição das coordenadas X 
    dos cavalos no frame. Quanto maior a variação no eixo X, maior será o fator de correção.
    """
    # Extrair as coordenadas X dos centroides
    x_values = [centroid[0] for centroid in centroids]

    # Calcular a variação nas coordenadas X (quanto mais espalhados, maior a inclinação)
    x_variance = np.var(x_values)

    # Normalizar a variação pela largura do frame para determinar o fator de correção
    correction_factor = (x_variance / frame_width) * max_correction

    # Garantir que o fator de correção não ultrapasse o valor máximo permitido
    correction_factor = min(correction_factor, max_correction)

    return correction_factor


def correct_for_camera_angle(x, y, frame_width, angle_correction_factor):
    """
    Aplica uma correção na coordenada Y com base na posição X e no fator de correção de ângulo.
    """
    corrected_y = y - (x / frame_width) * angle_correction_factor
    return corrected_y
