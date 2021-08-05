# Code for logging whether the gas, brake, or neither is being pressed
from time import sleep
import keyboard as key

# Variable for easy file naming
test_num = 1
# Key for starting and ending the logging
start_key = "p"

key.wait(start_key)
# Open a file to record all the key presses
with open(f"test_{test_num}-gas.txt", "w+") as file:
    while True: # w (gas) is recorded as a 2, s (brake) is recorded as a 1, and nothing is recorded as a 0
        if key.is_pressed('w'):
            file.write('2')
        elif key.is_pressed('s'):
            file.write('1')
        elif key.is_pressed('p'):
            break
        else:
            file.write('0')
        sleep(1//10)
