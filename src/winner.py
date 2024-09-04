import numpy as np


def determine_winner_based_on_speed(scene):
    """
    Determina o vencedor com base na média das velocidades dos competidores.
    Agora utiliza o array de velocidades acumuladas para cada competidor.
    """
    # Obter as médias de velocidade de cada competidor
    speeds = [np.mean(info['SPEEDS']) for info in scene.values()]

    # O vencedor é aquele com a maior média de velocidade
    winner_id = np.argmax(speeds)

    return winner_id


def determine_winner_in_scene(track_history):
    """
    Determina o vencedor de cada cena com base na média das velocidades.
    """
    for scene in track_history:
        # Determinar o vencedor com base na média das velocidades
        winner_id = determine_winner_based_on_speed(scene)

        # Marcar o vencedor na cena
        for idx, (rider_id, info) in enumerate(scene.items()):
            info['WINNER'] = 1 if idx == winner_id else 0
