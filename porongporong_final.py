
# 배리어 프리 영화 제작 인공지능_kusitms

## 1. 영상 다운 및 음원 추출
#### import, 영상 주소 입력, 함수 선언
from flask import Flask, render_template
from werkzeug.utils import secure_filename
from flask import request
app = Flask(__name__)

import pytube
import os
from google.cloud import storage
import json
import io
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
from google.cloud.speech_v1 import types
import subprocess
from pydub.utils import mediainfo
import subprocess
import math
import datetime
import srt

#딥러닝 라이브러리
import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from speakerfeatures import extract_features
import time
import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
import warnings
warnings.filterwarnings("ignore")
import logging

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\wh\\Desktop\\stt-test-294708-46be58928ce1.json"

BUCKET_NAME = "bongmin-bucket" # update this with your bucket name
link="https://www.youtube.com/watch?v=aWQg8AE2Frk"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['video']
        f.save(secure_filename(f.filename))

        def upload_blob(bucket_name, source_file_name, destination_blob_name):
            """Uploads a file to the bucket."""
            # bucket_name = "your-bucket-name"
            # source_file_name = "local/path/to/file"
            # destination_blob_name = "storage-object-name"

            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)

            blob.upload_from_filename(source_file_name)

            print(
                "File {} uploaded to {}.".format(
                    source_file_name, destination_blob_name
                )
            )


        def download_video(link):
            try: 
                #object creation using YouTube which was imported in the beginning 
                yt = pytube.YouTube(link) 
            except: 
                print("Connection Error") #to handle exception 
            video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()

            # rename the path
            new_path = video_path.split('/')
            new_filename = f"video.mp4"
            new_path[-1]= new_filename
            new_path='/'.join(new_path)
            os.rename(video_path, new_path)

            return new_path


        def video_info(video_filepath):
            """ this function returns number of channels, bit rate, and sample rate of the video"""

            video_data = mediainfo(video_filepath)
            channels = video_data["channels"]
            bit_rate = video_data["bit_rate"]
            sample_rate = video_data["sample_rate"]

            return channels, bit_rate, sample_rate


        def video_to_audio(video_filepath, audio_filename, video_channels, video_bit_rate, video_sample_rate):
            command = f"ffmpeg -i {video_filepath} -b:a {video_bit_rate} -ac {video_channels} -ar {video_sample_rate} -vn {audio_filename}"
            subprocess.call(command, shell=True)
            blob_name = f"audios/{audio_filename}"
            upload_blob(BUCKET_NAME, audio_filename, blob_name)
            return blob_name


        def long_running_recognize(storage_uri, channels, sample_rate):

            client = speech_v1.SpeechClient()

            config = {
                "language_code": "en-US",
                "sample_rate_hertz": int(sample_rate),
                "encoding": enums.RecognitionConfig.AudioEncoding.LINEAR16,
                "audio_channel_count": int(channels),
                "enable_word_time_offsets": True,
                "model": "video",
                "enable_automatic_punctuation":True
            }
            audio = {"uri": storage_uri}

            operation = client.long_running_recognize(config, audio)

            print(u"Waiting for operation to complete...")
            response = operation.result()
            return response


        def subtitle_generation(speech_to_text_response, bin_size=3):
            """We define a bin of time period to display the words in sync with audio. 
            Here, bin_size = 3 means each bin is of 3 secs. 
            All the words in the interval of 3 secs in result will be grouped togather."""
            transcriptions = []
            index = 0

            for result in response.results:
                try:
                    if result.alternatives[0].words[0].start_time.seconds:
                        # bin start -> for first word of result
                        start_sec = result.alternatives[0].words[0].start_time.seconds 
                        start_microsec = result.alternatives[0].words[0].start_time.nanos * 0.001
                    else:
                        # bin start -> For First word of response
                        start_sec = 0
                        start_microsec = 0 
                    end_sec = start_sec + bin_size # bin end sec

                    # for last word of result
                    last_word_end_sec = result.alternatives[0].words[-1].end_time.seconds
                    last_word_end_microsec = result.alternatives[0].words[-1].end_time.nanos * 0.001

                    # bin transcript
                    transcript = result.alternatives[0].words[0].word

                    index += 1 # subtitle index

                    for i in range(len(result.alternatives[0].words) - 1):
                        try:
                            word = result.alternatives[0].words[i + 1].word
                            word_start_sec = result.alternatives[0].words[i + 1].start_time.seconds
                            word_start_microsec = result.alternatives[0].words[i + 1].start_time.nanos * 0.001 # 0.001 to convert nana -> micro
                            word_end_sec = result.alternatives[0].words[i + 1].end_time.seconds
                            word_end_microsec = result.alternatives[0].words[i + 1].end_time.nanos * 0.001

                            if word_end_sec < end_sec:
                                transcript = transcript + " " + word
                            else:
                                previous_word_end_sec = result.alternatives[0].words[i].end_time.seconds
                                previous_word_end_microsec = result.alternatives[0].words[i].end_time.nanos * 0.001

                                # append bin transcript
                                transcriptions.append(srt.Subtitle(index, datetime.timedelta(0, start_sec, start_microsec), datetime.timedelta(0, previous_word_end_sec, previous_word_end_microsec), transcript))

                                # reset bin parameters
                                start_sec = word_start_sec
                                start_microsec = word_start_microsec
                                end_sec = start_sec + bin_size
                                transcript = result.alternatives[0].words[i + 1].word

                                index += 1
                        except IndexError:
                            pass
                    # append transcript of last transcript in bin
                    transcriptions.append(srt.Subtitle(index, datetime.timedelta(0, start_sec, start_microsec), datetime.timedelta(0, last_word_end_sec, last_word_end_microsec), transcript))
                    index += 1
                except IndexError:
                    pass
                
            # turn transcription list into subtitles
            subtitles = srt.compose(transcriptions)
            return subtitles

        ## 2. 자막 추출 및 음원 분할
        #video_path=download_video(link)
        video_path='video.mp4'
        channels, bit_rate, sample_rate = video_info(video_path)

        #### 자막 추출 및 확장자 변환(.srt --> .vtt)
        #blob_name=video_to_audio(video_path, "audio.wav", channels, bit_rate, sample_rate)
        blob_name=video_to_audio(video_path, "subtitles.wav", channels, bit_rate, sample_rate)

        #### 자막 추출 및 확장자 변환(.srt --> .vtt)
        gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
        response=long_running_recognize(gcs_uri, channels, sample_rate)
        subtitles= subtitle_generation(response)
        with open("subtitles.srt", "w") as f:
            f.write(subtitles)

        ######################################################!python srt2vtt.py subtitles.srt
        os.system('python srt2vtt.py subtitles.srt')

        #### 음원 분할   
        #이거 똑같은 거 두번 반복하는건가여..?
        ######################################################!python main.py  --input=C:\Users\wh\speech\Generate-SRT-File-using-Google-Cloud-s-Speech-to-Text-API 
        os.system(r'python main.py  --input=C:\Users\wh\speech\Generate-SRT-File-using-Google-Cloud-s-Speech-to-Text-API')

        ## 3. 화자 인식 (mfcc, gmm)  
        #### test_file = 분할된 음원 파일의 경로가 명시된 텍스트 파일
        #### modelpath = 화자별 .gmm모델이 저장되어있는 폴더
        #### input : .wav 파일, output : 음원 별 화자 (speakers[winner])
        #import os
        #import pickle as cPickle
        #import numpy as np
        #from scipy.io.wavfile import read
        #from speakerfeatures import extract_features
        #
        #import time
        #import numpy as np
        #from sklearn import preprocessing
        #import python_speech_features as mfcc
        #import warnings
        #warnings.filterwarnings("ignore")
        #
        #import logging

        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)

        # 함수 선언
        def calculate_delta(array):
            """Calculate and returns the delta of given feature vector matrix"""

            rows,cols = array.shape
            deltas = np.zeros((rows,20))
            N = 2
            for i in range(rows):
                index = []
                j = 1
                while j <= N:
                    if i-j < 0:
                        first = 0
                    else:
                        first = i-j
                    if i+j > rows -1:
                        second = rows -1
                    else:
                        second = i+j
                    index.append((second,first))
                    j+=1
                deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
            return deltas

        def extract_features(audio,rate):
            """extract 20 dim mfcc features from an audio, performs CMS and combines 
            delta to make it 40 dim feature vector"""    

            mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True)

            mfcc_feat = preprocessing.scale(mfcc_feat)
            delta = calculate_delta(mfcc_feat)
            combined = np.hstack((mfcc_feat,delta)) 
            return combined



        #모델 적용

        source   = "dst\\" 

        test_file = "test.txt"             
        file_paths = open(test_file,'r')



        modelpath = "train_models\\"       
        gmm_files = [os.path.join(modelpath,fname) for fname in 
                      os.listdir(modelpath) if fname.endswith('.gmm')]
        models    = [cPickle.load(open(fname,'rb'), encoding='utf-8') for fname in gmm_files]
        speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname 
                      in gmm_files]


        output_speakers = []

        for path in file_paths:   
            path = path.strip()   
            #print (path)
            #sr,audio = read(source + path)
            sr,audio = read(path)
            vector   = extract_features(audio,sr)

            log_likelihood = np.zeros(len(models)) 

            for i in range(len(models)):
                gmm    = models[i]         #checking with each model one by one
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()

            winner = np.argmax(log_likelihood)
            output_speakers.append(speakers[winner])
            #print ("\tdetected as - ", speakers[winner])

        ## 4. 자막(.srt)에 화자 추가
        #### output_speakers에 저장되어 있는 화자 정보 자막에 추가
        files = open("subtitles.srt",'r', encoding='UTF8')
        lines = files.readlines()
        files.close()

        with open("subtitles.srt", "w") as f:
            count = 0
            for i in range(len(lines)):
                if((i-2)%4 == 0):
                    lines[i] = "( " + output_speakers[count] + " ) " + lines[i]
                    count+=1
                f.write(lines[i])
            f.close()


        ## 5. 영상과 자막 합치기
        ######################################################!ffmpeg -i subtitles.srt subtitles.ass
        os.system('ffmpeg -i subtitles.srt subtitles.ass')
        ######################################################!ffmpeg -i video.mp4 -vf ass=subtitles.ass porongporong.mp4
        os.system('ffmpeg -i video.mp4 -vf ass=subtitles.ass porongporong.mp4')

        return render_template('check.html')


if __name__ =='__main__':
        #app.run(host='0.0.0.0', port=8080, debug = True)
    app.run()