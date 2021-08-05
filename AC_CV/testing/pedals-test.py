import keyboard as key
from time import sleep

# Define a key to tell the program to start running
start_key = "p"
most_recent = 0

key.wait(start_key)
with open("../test_model/test-direction.txt", 'r+') as file:
    commands = file.read()
    for command in commands:
        if command == '2':
            if not key.is_pressed('a'):
                key.press('a')
            if key.is_pressed('d'):
                key.release('d')
            most_recent = 2
        elif command == '1':
            if not key.is_pressed('d'):
                key.press('d')
            if key.is_pressed('a'):
                key.release('a')
            most_recent = 1
        else:
            if key.is_pressed('d'):
                key.release('d')
            if key.is_pressed('a'):
                key.release('a')
            most_recent = 0
        sleep(0.15)