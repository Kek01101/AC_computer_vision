# Code for logging whether the gas, brake, or neither is being pressed
from time import sleep
import keyboard as key

# Variable for easy file naming
test_num = 10
# Key for starting and ending the logging
start_key = "p"

key.wait(start_key)
# Open a file to record all the key presses
with open(f"data_{test_num}-direction.txt", "w+") as f1, open(f"data_{test_num}-gas.txt", "w+") as f2:
    sleep(2)
    while True: # a (left) is recorded as a 2, d (right) is recorded as a 1, and nothing is recorded as a 0
        if key.is_pressed('a'):
            f1.write('2')
        elif key.is_pressed('d'):
            f1.write('1')
        else:
            f1.write('0')
        if key.is_pressed('w'):
            f2.write('2')
        elif key.is_pressed('s'):
            f2.write('1')
        else:
            f2.write('0')
        if key.is_pressed('p'):
            break
        sleep(0.08)
