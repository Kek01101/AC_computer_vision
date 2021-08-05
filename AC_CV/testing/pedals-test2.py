import keyboard as key
from time import sleep

# Define a key to tell the program to start running
start_key = "p"

key.wait(start_key)
with open("../test_model/test-gas.txt", 'r+') as file:
    commands = file.read()
    for command in commands:
        if command == '2':
            key.send('w')
        elif command == '1':
            key.send('s')
        sleep(1//10)