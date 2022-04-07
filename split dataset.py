import os
import os.path
import random
from PIL import Image
from my_image_folder import is_image_file

def oversample_num(wind): 
    if wind < 60:
        return 1
    if wind < 80:
        return 1 + random.randint(0,1)
    if wind < 100:
        return 1 + random.randint(0,2)
    return 1 + random.randint(0,10)

def save_file(f,fname,f_root):
    wind = int(fname.split('_')[2])
    
    if oversample :
        cps = oversample_num(wind)
    else :
        cps = 1

    global count
    count = count + cps
    
    for i in range(0,cps):
        temp = fname.split('.')
        temp[0] = temp[0]+'_'+str(i)
        new_fname = temp[0]+'.'+temp[1]
        f.save(f_root+new_fname)

def if_match(f1,f2): 
    tname1 = f1.split('_')
    tname2 = f2.split('_')
    
    if tname1[0]!=tname2[0]:
        return False
    
    date1 = tname1[1]
    date2 = tname2[1]
    h1 = date1[len(date1)-1]
    h2 = date2[len(date2)-1]
 
    if (h1=='0' and h2=='6')or(h1=='6' and h2=='2')or(h1=='2' and h2=='8')or(h1=='8' and h2=='0'):
        return True
    else :
        return False

def cut_pics(p): 
    box = (128,128,384,384)
    p = p.crop(box)
    return p


def create_sample(source_dir,fname_1,fname_2,target_dir): 

    complete_fname_1 = os.path.join(root,fnames[i-1])
    complete_fname_2 = os.path.join(root,fnames[i])

    if not(is_image_file(complete_fname_1) and is_image_file(complete_fname_2)):
        return 'Not image file: ',complete_fname_1,complete_fname_2
    img_2 = Image.open(complete_fname_2)
    img_2 = cut_pics(img_2)
    save_file(img_2,fname_2,target_dir)


if __name__ == '__main__':

    path_ = os.path.abspath('/content/drive/MyDrive/TCIE')
    raw_dir = path_ + '/Datset/'

    train_root = path_ + '/train_set/'
    if not os.path.exists(train_root):
        os.mkdir(train_root)

    test_root = path_ + '/test_set/'
    if not os.path.exists(test_root):
        os.mkdir(test_root)

    global count, oversample
    count = 0
    oversample = True

    for root, _, fnames in sorted(os.walk(raw_dir)):

        fnames = sorted(fnames)
        boundary = int(len(fnames)*0.8)
        print("Boundary:",boundary);

    for i in range(1,boundary): 

	    info = create_sample(root,fnames[i-1],fnames[i],train_root)
	    if info:
                print(info)

                if count > 30000 :
                    print('Exceed the upper limit of a single file.')
                break

	    if i % 100 == 99 :
	        print('have processed ',i+1,' files.')

    print('items in train set: ',count)
    count = 0

    for i in range(boundary,len(fnames)): 

        info = create_sample(root,fnames[i-1],fnames[i],test_root)
        if info:
            print(info)

            if count > 30000 :
                print ('Exceed the upper limit of a single file.')
            break

        if i % 100 == 99 :
	        print('have processed ',i+1,' files.')

        print('items in test set: ',count)
