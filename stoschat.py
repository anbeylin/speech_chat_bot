
from helper import SpeechToAnswer
import keyboard

def main():
    while True:
        SpeechToAnswer()
        if keyboard.is_pressed('esc'):
            print("Exiting...")
            break

if __name__ == "__main__":
    main()
