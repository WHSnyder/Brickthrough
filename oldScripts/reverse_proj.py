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
    
    
random.seed()

for x in range(20000):
    
    i = random.randint(0,2)
    
    piece = objects[i]
    
    for y in range(3):
        
        if (y == i):
            objects[y].location = (0,0,0)
        else:
            objects[y].location = (1000000,1000000,1000000)
    
    angle = random.randint(0, 180)
    piece.rotation_euler.rotate_axis("Z", math.radians(angle))
    
    objects[i].location = (random.randint(0,3), random.randint(0,3), 0) 
    
    
    camera.location = (0, 0, random.randint(4,20))
    
    
    
    
    scene.render.resolution_x = 128
    scene.render.resolution_y = 128
    scene.render.resolution_percentage = 100
            
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = "/Users/will/projects/legoproj/training_images/test" + str(x) + ".png"
    bpy.ops.render.render(write_still = 1)
    
    depth = 
    
    labels += (str(x) + " " + piece.name + " " + (piece.location-camera.location).x + " " + pi + "\n")


print("Generated " + str(x+1) + " images in " + str(float(millis() - timestart)/1000.0) + " seconds") 

text_file = open("/Users/will/projects/legoproj/training_images/labels.txt", "w")
text_file.write(labels)
text_file.close()    