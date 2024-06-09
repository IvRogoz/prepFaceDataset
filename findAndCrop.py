import os
import cv2
import sys
from os import listdir
from os.path import isfile, join
from tkinter.filedialog import askdirectory
import tkinter as tk
import shutil

from facenet_pytorch import MTCNN

mtcnn = MTCNN(keep_all=True, post_process=False, device='cuda:0')

root = tk.Tk()
root.overrideredirect(1)
root.withdraw()

def update_progress(progress, total, found):
    filled_length = int(round(100 * progress / float(total)))
    sys.stdout.write('\r [\033[1;34mPROGRESS\033[0;0m] [\033[0;32m{0}\033[0;0m]:{1}% : Found:{2}'.format('#' * int(filled_length/5), filled_length, found))
    if progress == total:sys.stdout.write('\n')
    sys.stdout.flush()

def test_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("created folder : ", dir)
    else:
        print(dir, "folder already exists.")

directory = askdirectory()
cropped_dir = join(directory, './cropped/')
nop_dir = join(directory, './nop/')

test_dir(cropped_dir)
test_dir(nop_dir)

onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f)) & f.endswith(".jpg")]
i = 0
nop = 0
for n in range(0, len(onlyfiles)):
    current_file = join(directory, onlyfiles[n])
    try:
        image = cv2.imread(current_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        margin = int(min(height, width) * 0.1)  # 10% of the smaller dimension as margin

        # Detect faces in the image
        boxes, probs, _ = mtcnn.detect(gray, landmarks=True)
        count = 0
        for index, person in enumerate(boxes):
            person = person.astype(int)
            x, y, x1, y1 = person
            if probs[index] > 0.95 and (x1-x) > 50:
                count += 1
                # Apply margin and ensure the full image is visible
                x = max(x - margin, 0)
                y = max(y - margin, 0)
                x1 = min(x1 + margin, width)
                y1 = min(y1 + margin, height)
                face = image[y:y1, x:x1]
                resized_face = cv2.resize(face, (512, 512))
                face_filename = f"{onlyfiles[n].split('.')[0]}_face{index}.jpg"
                cv2.imwrite(join(cropped_dir, face_filename), resized_face)
        if count > 0:
            i += 1
        else:
            shutil.move(current_file, join(nop_dir, onlyfiles[n]))
            nop += 1
        update_progress(n, len(onlyfiles), count)
    except Exception as e:
        # print()
        # print(e,"<>",current_file)
        shutil.move(current_file, join(nop_dir, onlyfiles[n]))
        nop += 1

print()
print("Moved and cropped:", i)
print("Errored:", nop)
