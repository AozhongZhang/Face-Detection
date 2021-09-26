from PIL import Image

image = Image.open("./precrop/yuanshi/32-1.jpg")
print(image)
# cropped = image.crop((935,480,985,525))
cropped = image.crop((400,200,1480,880))
cropped.save("./precrop/chuli/1_crop1.jpg")