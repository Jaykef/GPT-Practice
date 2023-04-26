from neuralintents import GenericAssistant
import speech_recognition
import pyttsx3 as tts
import sys

recognizer = speech_recognition.Recognizer()

speaker = tts.init()
speaker.setProperty('rate', 150)
voices = speaker.getProperty('voices')
speaker.setProperty('voice', voices[40].id)

assignment_list = ['周一Python实践', '周二Python实践', '周三Python实践']

def new_homework():

    global recognizer

    speaker.say("你想在你的作业列表中添加什么？")
    speaker.runAndWait()

    done = False

    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)

                note = recognizer.recognize_google(audio)
                note = note.lower()

                speaker.say("请选文件名!")
                speaker.runAndWait()

                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)

                filename = recognizer.recognize_google(audio)
                filename = filename.lower()

            with open(filename, 'w') as f:
                f.write(note)
                done = True
                speaker.say(f"我成功加了你的{filename}作业")
                speaker.runAndWait()

        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            speaker.say("对不起，我没有得到， 请再说一遍")
            speaker.runAndWait()


def add_homework():

    global recognizer

    speaker.say("加什么作业?")
    speaker.runAndWait()

    done = False

    while not done:
        try:
            with speech_recognition.Microphone() as mic:

                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)

                item = recognizer.recognize_google(audio)
                item = item.lower()

                assignment_list.append(item)
                done = True

                speaker.say(f"我添加了{item}在你的作业列表！")
                speaker.runAndWait()

        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            speaker.say("对不起，我没有得到， 请再说一遍")
            speaker.runAndWait()

def show_homework():
    speaker.say("好的，这是你的作业清单")
    for item in new_homework:
        speaker.say(item)
    speaker.runAndWait()

def greeting():
    speaker.say("你好苏杰， 你需要什么帮助的？")
    speaker.runAndWait()

def quit():
    speaker.say("好的, 再见苏杰")
    speaker.runAndWait()
    sys.exit(0)


mappings = {
    "greeting": greeting,
    "new_homework": new_homework,
    "add_homework": add_homework,
    "show_homework": show_homework,
    "exit": quit
}

assistant = GenericAssistant('intents-cn.json', intent_methods=mappings)
assistant.train_model()

while True:

    try:
        with speech_recognition.Microphone() as mic:

            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)

            message = recognizer.recognize_google(audio)
            message = message.lower()

        assistant.request(message)

    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()

