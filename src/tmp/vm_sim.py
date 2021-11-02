import os
import time

txt_path = '/nfs/rtsan01/RT-Temp/TomBruijnen/machine_flip_angles.txt'

while True:
    if os.path.isfile(txt_path):
        os.remove(txt_path)
        print("Removed txt file")
    else:
        print("Waiting on txt file...")
        time.sleep(0.1)