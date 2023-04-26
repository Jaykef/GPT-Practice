from neuralintents import GenericAssistant
import speech_recognition
import pyttsx3 as tts
import sys

recognizer = speech_recognition.Recognizer()

speaker = tts.init()
speaker.setProperty('rate', 150)

assignment_list = ['Python Lab Week 1', 'Week 2 Python Lab', 'Week 3 Python assignment']

def create_homework():

    global recognizer

    speaker.say("What do you want me to write onto your assignment list?")
    speaker.runAndWait()

    done = False

    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.02)
                audio = recognizer.listen(mic)

                note = recognizer.recognize_google(audio)
                note = note.lower()

                speaker.say("Please choose a file name!")
                speaker.runAndWait()

                recognizer.adjust_for_ambient_noise(mic, duration=0.02)
                audio = recognizer.listen(mic)

                filename = recognizer.recognize_google(audio)
                filename = filename.lower()

            with open(filename, 'w') as f:
                f.write(note)
                done = True
                speaker.say(f"I successfully created {filename} list")
                speaker.runAndWait()

        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            speaker.say("I did not understand you, please say again!")
            speaker.runAndWait()


def add_homework():

    global recognizer

    speaker.say("What do you want to add to your assignment list?")
    speaker.runAndWait()

    done = False

    while not done:
        try:
            with speech_recognition.Microphone() as mic:

                recognizer.adjust_for_ambient_noise(mic, duration=0.02)
                audio = recognizer.listen(mic)

                item = recognizer.recognize_google(audio)
                item = item.lower()

                assignment_list.append(item)
                done = True

                speaker.say(f"I added {item} to your assignment list Jay!")
                speaker.runAndWait()

        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            speaker.say("I did not understand you, please say again!")
            speaker.runAndWait()

def show_homework():
    speaker.say("Okay boss, here are your assignment for the week")
    for item in assignment_list:
        speaker.say(item)
    speaker.runAndWait()

def greeting():
    speaker.say("What's up Jay, what can I do for you?")
    speaker.runAndWait()

def quit():
    speaker.say("okay, bye Jay")
    speaker.runAndWait()
    sys.exit(0)


mappings = {
    "greeting": greeting,
    "create_homework": create_homework,
    "add_homework": add_homework,
    "show_homework": show_homework,
    "exit": quit
}

assistant = GenericAssistant('intents-en.json', intent_methods=mappings)
assistant.train_model()

while True:

    try:
        with speech_recognition.Microphone() as mic:

            recognizer.adjust_for_ambient_noise(mic, duration=0.02)
            audio = recognizer.listen(mic)

            message = recognizer.recognize_google(audio)
            message = message.lower()

        assistant.request(message)

    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()

