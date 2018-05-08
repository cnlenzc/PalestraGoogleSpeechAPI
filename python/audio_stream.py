# -*- coding: UTF-8 -*-
import pyaudio
import wave
import io
import time
import collections
from datetime import datetime
import threading
import sys
import traceback
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import grpc
from enum import Enum, unique
import logging

logging.basicConfig(format='%(asctime)s [%(threadName)s %(module)s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


FORMAT = pyaudio.paInt16
ENCODING = enums.RecognitionConfig.AudioEncoding.LINEAR16
SAMP_WIDTH = 2
CHANNELS = 1
RATE = 16000
FRAMES_PER_BUFFER = 800
MAX_RECORD_SECONDS = 10.0
MAX_SECONDS_AFTER_SILENC = 3.0
MAX_SIZE_STREAM = 500


@unique
class ErrMsg(Enum):
    OK                  = ''
    SemInternet         = 'Não há conexão com a Internet'
    FalhaComunicacao    = 'Falha na comunicação com nossos serviços na internet'
    NaoEntendi          = 'Não entendi'
    NaoEntendiFala      = 'Não entendi o que você disse'
    NaoOuvi             = 'Não ouvi o que você disse'

OK = ErrMsg.OK


class ErrorRaise(Exception):
    def __init__(self, cod_err):
        super(ErrorRaise, self).__init__("(%s) %s" % (cod_err, cod_err.value))
        self.cod_err = cod_err


def print_except(name):
    logger.error("except (%s): %s" % (name, repr(sys.exc_info()[1])))
    traceback.print_exception(*sys.exc_info(), limit=20)


class AudioDataStream(io.BufferedIOBase):

    def __init__(self):
        self._content = collections.deque(maxlen=MAX_SIZE_STREAM)
        self._header = b''
        self._pos = -1
        self.is_sending = False
        self._write_header()

    def _write_header(self):
        logger.debug('write header')
        wave_file = wave.Wave_write(self)
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(SAMP_WIDTH)
        wave_file.setframerate(RATE)
        wave_file.writeframes(b'')
        wave_file.close()

    def write_file_wave(self, file_name):
        logger.debug('write_file_wave')
        wave_file = wave.open(file_name, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(SAMP_WIDTH)
        wave_file.setframerate(RATE)
        wave_file.writeframes(b''.join(list(self._content)[1:]))
        wave_file.close()

    def write(self, data):
        self._header += data
        # logger.info('write chunk=%s len=%s' % (len(data), self.tell()))

    def append(self, data):
        # logger.info('append chunk=%s len=%s pos=%s' % (len(data), self.tell(), self._pos))
        if self.is_sending:
            self._content.append(data)

    def read(self, chunk_size):
        if not self.is_sending:
            return None

        if self._pos == -1:
            self._pos = 0
            return self._header

        for temp in [0.04, 0.07, 0.1, 0.0]:
            # logger.info('read chunk=%s len=%s pos=%s temp=%s' % (chunk_size, self.tell(), self._pos, temp))
            if self._pos < self.tell():
                ret = self._content[self._pos]
                self._pos += 1
                if chunk_size != len(ret):
                    logger.warning("chunk_size != buffer_size (chunk=%s buffer=%s)" % (chunk_size, len(ret)))
                time.sleep(0.03)
                return ret
            elif temp == 0.0:
                self.stop_send()
                return None
            else:
                time.sleep(temp)


    def read_all(self):
        return b''.join(list(self._content))

    def start_send(self):
        # logger.info('start_send')
        self._pos = -1
        self.is_sending = True

    def stop_send(self):
        # logger.info('stop_send')
        self.is_sending = False

    def has_delay(self):
        # indica se o atrazo na leitura é superior a 100 chunks (equivalente a 5 segundos)
        return self.tell() - self._pos > 100

    def tell(self):
        return len(self._content)


class RecognizeThr(threading.Thread):

    logger.debug("Create google cloud speech client")
    speech_client = speech.SpeechClient()
    stream = None
    thread = None
    thread_num = 1


    def __init__ (self):
        logger.debug('Initializing RecognizeThr')
        threading.Thread.__init__(self, name="Rgz-Thread-%s" % RecognizeThr.thread_num)
        RecognizeThr.thread_num += 1
        self.daemon = False
        self.text_transcript = None
        self.cod_err = None


    def _try_run(self):
        try:
            logger.debug("speech_client.sample")
            start_time = datetime.now()
            response = None
            audio_sample = RecognizeThr.speech_client.sample(
                stream=RecognizeThr.stream,
                encoding=ENCODING,
                sample_rate_hertz=RATE)

            logger.debug("Start recognize streaming")
            results = audio_sample.streaming_recognize(
                language_code='pt-BR',
                interim_results=False,
                single_utterance=True,
                max_alternatives=1)

            logger.debug("Analizing result")
            for result in results:
                if result.is_final:
                    response = result
                    self.text_transcript = response.transcript.encode('utf8')
                    self.cod_err = OK
                    confidence = int(response.confidence * 100.0 + 0.5) if response.confidence else 0
                    RecognizeThr.stream.stop_send()
                    logger.info("--> '%s' %2s%% - Elapsed: %s" % (
                        response.transcript, confidence, datetime.now() - start_time))
                    break

            if response:
                if confidence < 30:
                    self.cod_err = ErrMsg.NaoEntendiFala
            else:
                if not self.cod_err:
                    self.cod_err = ErrMsg.NaoEntendi

        except grpc._channel._Rendezvous as err:
            logger.error('except %s: %s' % (type(err), repr(err)))
            if repr(err).find('StatusCode.DEADLINE_EXCEEDED') != -1:
                self.cod_err = ErrMsg.SemInternet
            elif repr(err).find('StatusCode.UNAVAILABLE') != -1:
                logger.info("Retry google recognize - Elapsed: %s" % (datetime.now() - start_time))
                return True
            else:
                self.cod_err = ErrMsg.FalhaComunicacao

        except:
            print_except(__name__)
            self.cod_err = ErrMsg.FalhaComunicacao

        return False


    def run(self):
        logger.debug("Start recognize by google")
        start_time = datetime.now()

        RecognizeThr.stream = AudioDataStream()
        RecognizeThr.stream.start_send()

        if self._try_run():
            if self._try_run():
                self.cod_err = ErrMsg.FalhaRecVoz

        if self.cod_err != OK:
            logger.debug("No transcript - %s - Elapsed Time: %s" % (self.cod_err.value, datetime.now() - start_time))
        RecognizeThr.stream.stop_send()


    def get_transcript(self, timeout=MAX_RECORD_SECONDS):
        i = timeout
        sleep = 0.3
        while i > 0.0 and RecognizeThr.thread.is_alive():
            logger.debug("Say anything... %2s" % (i))
            RecognizeThr.thread.join(sleep)
            i -= sleep
            if RecognizeThr.stream.has_delay():
                self.cod_err = ErrMsg.SemInternet
                break

        if not self.cod_err:
            self.cod_err = ErrMsg.NaoOuvi

        RecognizeThr.stream.stop_send()
        RecognizeThr.thread.join(MAX_SECONDS_AFTER_SILENC)

        if self.cod_err != OK:
            logger.info("--> ?????? - %s" % (self.cod_err.value))
            raise ErrorRaise(self.cod_err)

        return self.text_transcript


if __name__ == '__main__':
    # exemplo para teste da classe

    def audio_callback(in_data, frame_count, time_info, status):
        # logger.info('Recording... frame_count=%s audio_len=%s' % (frame_count, RecognizeThr.stream.tell()))
        if RecognizeThr.stream:
            RecognizeThr.stream.append(in_data)
        play_data = chr(0) * len(in_data)
        return play_data, pyaudio.paContinue


    def start_recording():
        logger.debug("Start PyAudio")
        py_audio = pyaudio.PyAudio()
        audio_stream_in = py_audio.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        frames_per_buffer=FRAMES_PER_BUFFER,
                                        stream_callback=audio_callback)
        logger.info("Start recording")
        return py_audio, audio_stream_in


    def finish_recording(py_audio, audio_stream_in):
        logger.debug("Finishing PyAudio")
        audio_stream_in.stop_stream()
        audio_stream_in.close()
        py_audio.terminate()


    logger.debug("PP start recording PyAudio")
    py_audio, audio_stream_in = start_recording()

    logger.debug("PP start thread recognize google")
    RecognizeThr.thread = RecognizeThr()
    RecognizeThr.thread.start()

    logger.debug("PP wait for finishing recognize")
    try:
        text = RecognizeThr.thread.get_transcript()
        print(text)
    except ErrorRaise as err:
        print(err.cod_err.value)

    logger.info("Finish recording")
    finish_recording(py_audio, audio_stream_in)

    logger.debug("PP finished")
