import speech_recognition as sr

GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""./client_secret_872561412805-9enrums9i7isr2rnqa82m3foq9lonmfj.apps.googleusercontent.com.json"""

r = sr.Recognizer()
mic = sr.Microphone()

try:
    while True:
        with mic as source:
            print("Say Something!! :)")
            audio = r.listen(source)
        print("Google think you said: "+ r.recognize_google(audio, language='ko'))

except sr.UnknownValueError:
    print("Gogle could not understand audio")
except sr.RequestError as e:
    print("Coould not request results service: {0}".format(e))