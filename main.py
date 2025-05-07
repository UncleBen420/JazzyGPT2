import time
from src.jazz_master import JazzMaster
from src.stream import ScoreProcessor


if __name__ == "__main__":

    N_BAR_TO_PLAY = 20
    TEMPO = input("Tempo: ")
    try:
        TEMPO = int(TEMPO)
    except Exception:
        print('TEMPO put at 80bpm')
        TEMPO = 80

    if TEMPO > 120 or TEMPO < 60:
        print('TEMPO put at 80bpm')
        TEMPO = 80

    print('LOADING THE MODEL')
    jazz_masta = JazzMaster(quantization='Q4')
    print('INITIATING THE PERFORMANCE')
    sp = ScoreProcessor(jazz_masta,
                        n_bar_to_play=N_BAR_TO_PLAY,
                        tempo=TEMPO,
                        temp=0.3)

    print(f'Get Ready: {4}')
    time.sleep(1)
    print(f'Get Ready: {3}')
    time.sleep(1)
    print(f'Get Ready: {2}')
    time.sleep(1)
    print(f'Get Ready: {1}')
    time.sleep(1)
    sp.prepare()
    sp.run()

    time.sleep((60 / TEMPO) * 4 * N_BAR_TO_PLAY)
    print('THANKS FOR PLAYING, NOW MOVE ON')

    sp.stop()


