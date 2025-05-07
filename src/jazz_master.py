from llama_cpp import Llama
import miditok
import numpy as np
from symusic import Score
import warnings
import time
from tqdm import tqdm


class JazzMaster:
    def __init__(self, quantization='F16', overfitted=False,
                 cuda=True, verbose=False):

        if quantization == 'F16':
            model_name = "JazzyGPT2_V2-91M-F16.gguf"
        elif quantization == 'Q8':
            model_name = "JazzyGPT2_V2-91M-Q8.gguf"
        elif quantization == 'Q4':
            model_name = "JazzyGPT2_V2-91M-Q4.gguf"
        elif quantization == 'MAX':
            model_name = "JazzyGPT2-91M-Q4.gguf"
        else:
            raise Exception("Unkown quantisation type, only (F16, Q8 or Q4)")

        if overfitted:
            model_name = "JazzyGPT2_overfitted-91M-F16.gguf"
        
        if cuda:
            n_gpu_layers = -1
        else:
            n_gpu_layers = 0

        self.model = Llama.from_pretrained(
            repo_id="REMSLEGRAND/Jazzy_gpt2_v2",
            filename=model_name,
            n_gpu_layers=n_gpu_layers,
            n_ctx=1024,
            verbose=verbose
        )

        self.quant = quantization

        self.verbose = verbose

        self.tokenizer = miditok.MusicTokenizer.from_pretrained(
            'REMSLEGRAND/jazzy_gpt2')
        self.vocab = self.tokenizer.vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def _ids_to_string(self, ids):
        return " ".join(self.id_to_token[i]
                        for i in ids if i in self.id_to_token)

    def __call__(self, melody, temp=0.3):
        local_pred = [5, 2, 6]
        try:
            melody = self._ids_to_string(melody)

            output = self.model(
                melody,
                max_tokens=128,
                stop=["PAD_None"],
                temperature=temp,
                top_p=0.95,
                min_p=0.05,
                top_k=50
            )

            text = output['choices'][0]['text']
            if 'BOBA1_None' in text:
                if self.quant == 'MAX':
                    text = text.split('EOBA_None')[1]
                else:
                    text = text.split('EOBM_None')[1]
                local_pred = self.model.tokenize(text.encode("utf-8"),
                                                 add_bos=False,
                                                 special=True)
                local_pred[0] = 5
                local_pred += [6]
        except Exception as e:
            warnings.warn(f"Exception was catched during inference: {e}")

        return local_pred

    def predict_next_bar_score(self, score: Score, temp=0.3) -> Score:

        ids = self.tokenizer.encode(score).ids
        indices = np.where(np.array(ids) == 2)[0]

        # If there is more bars than 10
        if len(indices) >= 10:
            indices = indices[-9:]
            ids = ids[indices[0]:]
            indices -= indices[0]

        # Add the melody bar token
        id_correction = 0
        total_indices = len(indices) - 1
        for num_i, i in enumerate(indices):
            if i != 0:
                ids.insert(i + id_correction, 4)
                id_correction += 1
            ids.insert(i + id_correction,
                       self.tokenizer.vocab[
                           f'BOBM{- total_indices + num_i}_None'])
            id_correction += 1
        ids.append(4)

        # If there is more than 256 tokens, we crop at the
        # start of a bar
        if len(ids) > 256:
            idx = np.where(ids[-256:] == 4)[0]
            if idx.size > 0:
                ids = ids[-256 + idx[0] + 1:]
            else:
                ids = ids[-256:]
        ids.pop(-1)
        preds = self(melody=ids, temp=temp)

        return self.tokenizer.decode(preds)

    @staticmethod
    def get_latency(quantization='F16', cuda=True):
        model = JazzMaster(quantization=quantization, cuda=cuda)
        str_dummy_input = 'BOBM0_None Bar_None TimeSig_4/4 Position_0 Tempo_121.29 Position_277 Program_0 Pitch_68 Velocity_99 Duration_0.11.128 Program_0 Pitch_64 Velocity_91 Duration_0.11.128 Position_278 Program_0 Pitch_74 Velocity_79 Duration_0.12.128 Program_0 Pitch_76 Velocity_87 Duration_0.10.128 Position_385 Program_0 Pitch_69 Velocity_75 Duration_0.17.128 Program_0 Pitch_67 Velocity_83 Duration_0.12.128 Program_0 Pitch_72 Velocity_79 Duration_0.14.128 Position_386 Program_0 Pitch_63 Velocity_75 Duration_0.11.128 EOBM_None'
        times = []
        lenghts = []

        for i in tqdm(range(100), desc="Estimating Latency", unit="iter"):
            start_time = time.time()
            output = model.model(
                str_dummy_input,
                max_tokens=128,
                stop=["PAD_None"],
                temperature=0.3
            )
            end_time = time.time()

            times.append(end_time - start_time)
            text = output['choices'][0]['text']
            ids = model.model.tokenize(text.encode("utf-8"), add_bos=False,
                                       special=True)

            lenghts.append(len(ids))

            time.sleep(0.01)  # to add some variability

        return times, lenghts


if __name__ == "__main__":
    times, lenghts = JazzMaster.get_latency()

    print(f"Lenghts: Mean({np.mean(lenghts)}), STD({np.std(lenghts)}), Min({np.min(lenghts)}), Max({np.max(lenghts)})")

    print("CUDA Float16")
    print(f"Times: Mean({np.mean(times)}), STD({np.std(times)}), Min({np.min(times)}), Max({np.max(times)})")

    times, lenghts = JazzMaster.get_latency(quantization='Q8')
    print("CUDA Quantize Q8")
    print(f"Times: Mean({np.mean(times)}), STD({np.std(times)}), Min({np.min(times)}), Max({np.max(times)})")

    times, lenghts = JazzMaster.get_latency(quantization='Q4')
    print("CUDA Quantize Q4")
    print(f"Times: Mean({np.mean(times)}), STD({np.std(times)}), Min({np.min(times)}), Max({np.max(times)})")

    times, lenghts = JazzMaster.get_latency(cuda=False)
    print("CPU Float16")
    print(f"Times: Mean({np.mean(times)}), STD({np.std(times)}), Min({np.min(times)}), Max({np.max(times)})")

    times, lenghts = JazzMaster.get_latency(quantization='Q4', cuda=False)
    print("CPU Quantize")
    print(f"Times: Mean({np.mean(times)}), STD({np.std(times)}), Min({np.min(times)}), Max({np.max(times)})")
