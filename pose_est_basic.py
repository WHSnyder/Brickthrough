import bpy
import random
import math
import time



millis = lambda: int(round(time.time() * 1000))

timestart = millis()

scene = bpy.context.scene


pole = bpy.data.objects['Pole']
brick = bpy.data.objects['Brick']
wing = bpy.data.objects['Wing'] 

camera = bpy.data.objects['Camera']

objects = [pole, brick, wing]

mat = bpy.data.materials.new(name="Testmat")


r = 0.0#random.randrange(0, 10, 2)/10.0;
g = 0.0#random.randrange(0, 10, 2)/10.0;
b = 1.0#random.randrange(0, 10, 2)/10.0;
    
mat.diffuse_color = (r,g,b)




labels = ""





for piece in objects:
    
    piece.data.materials.append(mat)
    piece.data.materials[0] = mat 
    
    

for x in range(20000):
    
    piece = objects[0]
    
    angle = random.randint(0, 180)
    piece.rotation_euler.rotate_axis("Z", math.radians(angle))
    
    piece.location = (0, 0, 0) 
    
    
    camera.location = (0, 0, random.randint(10,30))
    
    scene.render.resolution_x = 128
    scene.render.resolution_y = 128
    scene.render.resolution_percentage = 100
            
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.filepath = "/Users/will/projects/legoproj/pose_basic_train/test" + str(x) + ".png"
    bpy.ops.render.render(write_still = 1)
    
    labels += (str(x) + " " + str(piece.rotation_euler.z) + "\n")


print("Generated " + str(x+1) + " images in " + str(float(millis() - timestart)/1000.0) + " seconds") 

text_file = open("/Users/will/projects/legoproj/pose_basic_train/labels.txt", "w")
text_file.write(labels)
text_file.close()    