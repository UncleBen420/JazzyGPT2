import sched
import time
import mido
import threading
from symusic import Score, Note, Track
from functools import partial
import fluidsynth

# Function to send MIDI messages
def send_midi_event(is_on, vel, pitch, channel, synth,):
    if is_on:
        synth.noteon(chan=channel, key=pitch, vel=vel)
        # msg = mido.Message('note_on',
        #                    note=pitch,
        #                    velocity=vel,
        #                    channel=channel)
    else:
        synth.noteoff(chan=channel, key=pitch)
    #     msg = mido.Message('note_off', note=pitch)
    # out.send(msg)

def predict_next_bar(sp):
    score = sp.score
    score = sp.jazz_master.predict_next_bar_score(score,
                                            sp.temperature)
    sp.add_score_to_performance(score)

def increment_quarter_counter(sp):
    print(sp.current_Q)
    sp.current_Q += 1

def callback_input(message, score_processor):

    # convert messages to symusic events
    current_time_sec = time.time() - score_processor.t_start
    current_time_ticks = mido.second2tick(current_time_sec,
                                          score_processor.tick_per_beat,
                                          score_processor.tempo)

    if message.type == "note_on":

        vel = message.velocity
        if score_processor.running_flag:
            score_processor.temp_note[message.note] = (current_time_ticks,
                                                       message.velocity)
        send_midi_event(True, vel, message.note, 0, score_processor.output_port)

    elif message.type == "note_off":
        if score_processor.running_flag:
            if message.note in score_processor.temp_note.keys():
                time_e, vel_e = score_processor.temp_note[message.note]
                del score_processor.temp_note[message.note]

                note = Note(time=time_e,
                            duration=current_time_ticks - time_e,
                            pitch=message.note, 
                            velocity=vel_e)

                score_processor.current_track.notes.append(note)
        send_midi_event(False, 0, message.note, 0, score_processor.output_port)

    elif message.is_cc(2): # temperature change
        val = message.value
        score_processor.temperature = round(val / 127 , 1)
        print("Temperature change: ", score_processor.temperature)

class ScoreProcessor:

    def __init__(self, jazz_master, n_bar_to_play=20, tempo=80, tpq=220, temp=0.3):
        self.output_names = mido.get_output_names()
        self.output_names = [name for name in self.output_names if 'FLUID' in name]
        print(self.output_names)
        self.n_bar_to_play = n_bar_to_play
        self.T0 = 0
        self.current_Q = 0
        self.current_T = 0
        self.t_start = 0
        self.temperature = temp
        self.tempo = mido.bpm2tempo(tempo, time_signature=(4, 4))
        # tpq is suppose to be tick per quarter but seems to be tick ber beat
        self.tick_per_beat = tpq  # score.tpq
        self.tick_per_quarter = self.tick_per_beat * 4
        self.one_tick_duration = mido.tick2second(1, self.tick_per_beat,
                                                  self.tempo)
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.scheduler_pred = sched.scheduler(time.time, time.sleep)
        self.running_flag = False
        self.thread = None
        self.thread_pred = None
        self.score = Score()
        self.score.tpq = self.tick_per_beat
        self.current_track = Track()
        self.temp_note = {}
        self.score.tracks.append(self.current_track)

        self.jazz_master = jazz_master

        # output port assume that there are 2 synths (piano on 1 and drum on 2)

        self.piano = fluidsynth.Synth(gain=.5, samplerate=48000)
        self.piano.start(driver='alsa', midi_driver='alsa_seq')
        sfid = self.piano.sfload("/home/orin/Documents/Jazzbox/Kawai_ES100_Mellow_Grand_1.sf2")
        self.piano.program_select(0, sfid, 0, 0)

        self.metronome = fluidsynth.Synth(gain=.5, samplerate=48000)
        self.metronome.start(driver='alsa', midi_driver='alsa_seq')
        # chan 9, sfont 1, bank 128, preset 0, Standard
        sfid = self.metronome.sfload("/home/orin/Documents/Jazzbox/Standard_Kit.sf2")
        self.metronome.program_select(9, sfid, 128, 0)

        self.output_port = self.piano
        self.output_port_met = self.metronome


        # self.output_port = mido.open_output(self.output_names[-2])
        # self.output_port_met = mido.open_output(self.output_names[-1])
        
        input_name = [name for name in mido.get_input_names() if 'MPK Mini Mk II' in name]
        print(mido.get_input_names())

        cllbck = partial(callback_input, score_processor = self)
        self.input_port = mido.open_input(input_name[0],
                                          callback=cllbck)

    def add_score_to_performance(self, score: Score, start_at_q=-1):
        score = score.sort()

        if start_at_q == -1:  # start at next quarter
            start = (self.current_Q + 1) * self.tick_per_quarter
        else:
            start = start_at_q * self.tick_per_quarter

        for track in score.tracks:
            for i, note in enumerate(track.notes):
                time_t = note.time
                start_sec = mido.tick2second(
                    start + time_t, self.tick_per_beat, self.tempo)

                velocity = note.velocity
                pitch = note.pitch
                duration = note.duration
                end_sec = mido.tick2second(
                    duration, self.tick_per_beat, self.tempo)

                delay = start_sec + self.t_start

                self.scheduler.enterabs(delay, 99, send_midi_event,
                                 (True, velocity, pitch, 0,
                                  self.output_port))
                self.scheduler.enterabs(delay, 99, send_midi_event,
                                 (False, velocity, pitch, 0,
                                  self.output_port))

    def prepare(self):
        for i in range(self.n_bar_to_play * 4):
            pitch = 56 if i % 4 == 0 else 54
            self.scheduler.enter(
                i * self.one_tick_duration * self.tick_per_beat, 50,
                send_midi_event, (True, 100, pitch, 9, self.output_port_met))
            
            if (i + 3) % 4 == 0 and i > 4: # each third beat of the quarter
                self.scheduler_pred.enter(
                    i * self.one_tick_duration * self.tick_per_beat, 99,
                    predict_next_bar,
                    (self,))

            elif i % 4 == 0 and i > 0:
                self.scheduler.enter(
                    i * self.one_tick_duration * self.tick_per_beat, 99,
                    increment_quarter_counter,
                    (self,))

    # run the predicton scheduler in another thread
    def _predict_worker(self):
        while True:
            self.scheduler_pred.run(blocking=False)
            time.sleep(self.one_tick_duration)

            if not self.running_flag:
                break

    def _run(self):
        self.t_start = time.time()
        while True:
            self.current_T += 1
            self.scheduler.run(blocking=False)
            time.sleep(self.one_tick_duration)

            if not self.running_flag:
                break

    def run(self):
        self.running_flag = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        self.thread_pred = threading.Thread(target=self._predict_worker)
        self.thread_pred.start()

    def stop(self):
        self.running_flag = False
        self.thread.join()
        self.thread_pred.join()