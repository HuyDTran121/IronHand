from pynput import keyboard
import pyaudio
import time
import wave
from rev_ai.speechrec import RevSpeechAPI
import speech_recognition as sr

CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = 'output.wav'
AMBIENT_FILENAME = 'ambient.wav'

p = pyaudio.PyAudio()
frames = []


class AudioTranscripter:
    def __init__(self):
        self.listener = MyListener()
        self.id = None
        self.google = GoogleTranscripter()
        self.listener.start()
        self.listener.stream.start_stream()
        self.start()
        time.sleep(1.5)
        self.stop(AMBIENT_FILENAME)
        file = sr.AudioFile(AMBIENT_FILENAME)
        with file as source:
            self.google.r.adjust_for_ambient_noise(source)
        self.rev = RevSpeechAPI('01XCwQ0XA0bDQsGJnjXD2nvoSrvOb9Ao9dxqmDYqgQnx5RJTKG0KMh-7rj2K4o0nG6kWvB7xXeEiZSjlfxKYv9UZ7ipiE')
        self.text_ready = False

    def start(self):
        self.listener.key_pressed = True
        frames.clear()

    def stop(self, filename=WAVE_OUTPUT_FILENAME):
        self.listener.key_pressed = None
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        if filename != AMBIENT_FILENAME:
            try:
                google_text = self.google.getText(filename)
                print(google_text)
                resp = self.rev.submit_job_local_file(filename)
                self.id = resp['id']
                return google_text
            except Exception:
                print("Couldn't Recognize")
                return None
        return None

    def returnRevText(self):
        if self.id is None:
            return None
        else:
            return self.rev.get_transcript(self.id, use_json=False)


class MyListener(keyboard.Listener):

    def __init__(self):
        super(MyListener, self).__init__(self.on_press, self.on_release)
        self.key_pressed = None

        self.stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.callback,
        )

    def on_press(self, key):
        if False:
            self.key_pressed = True

    def on_release(self, key):
        if False:
            self.key_pressed = False

    def callback(
            self,
            in_data,
            frame_count,
            time_info,
            status,
    ):
        if self.key_pressed:
            frames.append(in_data)
            return (in_data, pyaudio.paContinue)
        else:
            return (in_data, pyaudio.paContinue)


class GoogleTranscripter:
    def __init__(self):
        self.r = sr.Recognizer()

    def getText(self, filename):
        file = sr.AudioFile(filename)
        with file as source:
            audio = self.r.record(source)
            return self.r.recognize_google(audio)
