import os
from PIL import Image

path = "/home/paula/Doutoramento/imagesTiff/TIFF"

def parse_tif(file, path_name):
    img = Image.open(file)
    for i in range (img.n_frames):
        img.seek(i)
        img.save(path_name+'/'+'Block_%s.tif'%(i,))

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if file.endswith(".tif") | file.endswith(".TIF"):
            files.append(os.path.join(r, file))

for f in files:
    path_file = os.path.dirname(f)
    head, tail = os.path.split(f)
    new_path = tail.replace(".tif", "")
    createFolder(path + "/" + new_path)
    parse_tif(f, path+"/"+new_path)

print("FINISHED!")