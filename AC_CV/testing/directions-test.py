import keyboard as key
from time import sleep

# Define a key to tell the program to start running
start_key = "p"
most_recent = 0

key.wait(start_key)
with open("../test_model/test-gas.txt", 'r+') as file:
    commands = file.read()
    for command in commands:
        if command == '2':
            if not key.is_pressed('w'):
                key.press('w')
            if key.is_pressed('s'):
                key.release('s')
            most_recent = 2
        elif command == '1':
            if not key.is_pressed('s'):
                key.press('s')
            if key.is_pressed('w'):
                key.release('w')
            most_recent = 1
        else:
            if key.is_pressed('s'):
                key.release('s')
            if key.is_pressed('w'):
                key.release('w')
            most_recent = 0
        sleep(0.15)