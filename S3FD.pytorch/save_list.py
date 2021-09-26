
import os
import os.path
import natsort

# rootdir="/media/sdd/daguo/dataset_0121/images_VIS/"
# rootdir="/media/sdd/daguo/dataset_0121/faces_VIS_128/"
# rootdir="/media/sdd/daguo/dataset_0121/images_IR/"
# rootdir="/media/sdd/daguo/dataset_0121/faces_IR_128/"


# rootdir="/media/sdd/daguo/IR_1000_dataset/imges_p"
# rootdir="/media/sdd/daguo/IR_1000_dataset/VIS_images"

# savefile="/media/sdd/daguo/IR_1000_dataset/imges_p/all_IR_p_images.txt"
# savefile="/media/sdd/daguo/IR_1000_dataset/VIS_images/all_VIS_images.txt"


# rootdir="/media/sdd/daguo/IR_1000_dataset/IR_p_faces_128"
# savefile="/media/sdd/daguo/IR_1000_dataset/IR_p_faces_128/all_IR_faces_128.txt"

# rootdir="/media/sdd/daguo/IR_1000_dataset/IR_p_faces_256"
# savefile="/media/sdd/daguo/IR_1000_dataset/IR_p_faces_256/all_IR_faces_256.txt"

# rootdir="/media/sdd/daguo/IR_1000_dataset/VIS_faces_128"
# savefile="/media/sdd/daguo/IR_1000_dataset/VIS_faces_128/all_VIS_faces_128.txt"

# rootdir="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/IR_p_faces_256"
# savefile="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/IR_p_faces_256/all_IR_p_faces_frontal_256.txt"

# rootdir="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/VIS_faces_256"
# savefile="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/VIS_faces_256/all_VIS_faces_frontal_256.txt"

# rootdir="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/IR_p_images_aligned/"
# savefile="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/all_IR_p_images_frontal_aligned.txt"

# rootdir="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/IR_p_faces_aligned_256_cleaned/"
# savefile="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/all_IR_p_faces_frontal_aligned_cleaned_256.txt"

# rootdir="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/VIS_faces_aligned_256_cleaned/"
# savefile="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/all_VIS_faces_frontal_aligned_cleaned_256.txt"

# rootdir="/media/sdd/daguo/IR_1000_dataset/VIS_images"
# savefile="/media/sdd/daguo/IR_1000_dataset/all_VIS_images_240_after.txt"

rootdir="/media/sdd/daguo/IR_1000_dataset/IR_p_23_frontal_images"
savefile="/media/sdd/daguo/IR_1000_dataset/all_IR_p_23_frontal_que241.txt"

# rootdir="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/IR_p_images_aligned_cleaned/"
# savefile="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/all_IR_p_images_frontal_aligned_cleaned.txt"

# rootdir="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/VIS_images_aligned_cleaned/"
# savefile="/media/sdd/daguo/IR_1000_dataset/frontal_256_100/all_VIS_images_frontal_aligned_cleaned.txt"

# rootdir="/media/sdd/daguo/dataset_0127/IR_p_faces_128/"
# # rootdir="/media/sdd/daguo/dataset_0127/VIS_faces_128/"
# savefile="/media/sdd/daguo/dataset_0127/all_IR_p_faces_128.txt"
# # savefile="/media/sdd/daguo/dataset_0127/all_VIS_faces_128.txt"


# savefile="/media/sdd/daguo/dataset_0121/all_IR_imgs.txt"
# savefile="/media/sdd/daguo/dataset_0121/all_IR_imgs_128_frontal.txt"
# savefile="/media/sdd/daguo/dataset_0121/all_VIS_imgs.txt"
# savefile="/media/sdd/daguo/dataset_0121/all_VIS_imgs_128_frontal.txt"
out = open(savefile,'w')
imglist=[]
print ('start...')
for parent,dirnames,filenames in os.walk(rootdir):
    for filename in filenames:
        # if filename.strip().split('_')[-1] == 'IR.jpg': #IR
        if filename.strip().split('.')[1] == 'bmp': #IR
        # if filename.strip().split('.')[1] == 'jpg': #VIS
        # if filename.strip().split('_')[-1] == 'VIS.jpg':  #VIS
            imglist.append(str(os.path.join(parent,filename)))
sort_list = natsort.natsorted(imglist)
for i in range(len(sort_list)):
    #print(sort_list[int(i)])
    out.write(sort_list[i] + '\n') #+ ' ' + sort_list[i].split('/')[6] + '\n')
out.close()
print ('successful...')

# print ('start...')
# for parent,dirnames,filenames in os.walk(rootdir):
#     for filename in filenames:
#         # if filename.strip().split('_')[-1] == 'IR.jpg': #IR
#         # if filename.strip().split('.')[1] == 'bmp': #IR
#         if filename.strip().split('_')[-1] == 'VIS.jpg':  #VIS
#           if( (filename.strip().split('_')[1] != 'p4') and (filename.strip().split('_')[1] != 'p5') ):  #VIS
#                 imglist.append(str(os.path.join(parent,filename)))
# sort_list = natsort.natsorted(imglist)
# for i in range(len(sort_list)):
#     #print(sort_list[int(i)])
#     out.write(sort_list[i] + '\n') #+ ' ' + sort_list[i].split('/')[6] + '\n')
# out.close()
# print ('successful...')

# print ('start...')
# for parent,dirnames,filenames in os.walk(rootdir):
#     for filename in filenames:
#         if filename.strip().split('_')[-1] == 'IR.jpg': #IR
#         # if filename.strip().split('.')[1] == 'bmp': #IR
#         # if filename.strip().split('.')[1] == 'jpg': #VIS
#         # if filename.strip().split('_')[-1] == 'VIS.jpg':  #VIS
#         #   print(len(filename.strip().split('-')))
#           if( len(filename.strip().split('-')) == 2):  #IR
#                 imglist.append(str(os.path.join(parent,filename)))
# sort_list = natsort.natsorted(imglist)
# for i in range(len(sort_list)):
#     #print(sort_list[int(i)])
#     out.write(sort_list[i] + '\n') #+ ' ' + sort_list[i].split('/')[6] + '\n')
# out.close()
# print ('successful...')
