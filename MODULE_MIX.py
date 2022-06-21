import shutil, os
import random

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'folder created: {dir} \n')

def check_total_num_per_cat(dir, count, src_dir = None):
    files = os.listdir(dir)
    if len(files) == count:
        print(f'# files under {dir} matches {count}. \n')
    elif len(files) < count:
        diff = count - len(files)
        if src_dir != None:
            more_files = os.listdir(src_dir)
#            rand_idx = random.sample(range(0,count), diff)
            for i in range(diff):
                rand_idx = random.sample(range(0,count), 1)
                while more_files[rand_idx[0]] in dir:
                    rand_idx = random.sample(range(0,count), 1)
                src_file = f'{src_dir}/{more_files[rand_idx[0]]}'
                shutil.copy(src_file,dir)
                print(f'File {src_file} is copied to {dir}\n')
            print(f'{diff} images are added.')
    else:
        diff = len(files) - count
        rand_idx = random.sample(range(0,len(files)), diff)            
        for i in range(diff):
            os.remove(f'{dir}/{files[rand_idx[i]]}')
            print(f'File {files[rand_idx[i]]} is removed from {dir}\n')
    files_after = os.listdir(dir)
    print(f'The final number of images in {dir} is {len(files_after)}\n')
    print('*****************************\n')

def Mix_training_set_gen(src_mother_dirs, dest_mother_dir, object_names, purpose, training_set_name, num_per_cat, proportions):
    for object in object_names:
        for i in range(len(src_mother_dirs)):
            src_dir = f'{src_mother_dirs[i]}/{object}'
            check_dir(src_dir)
            src_dir_files = os.listdir(src_dir)
            num_src_img = len(src_dir_files)
            print(f'{num_src_img} in the source folder {src_dir}\n')
            if num_src_img == 0:
                print('Error: Nothing in the source directory.')
                return None

            dest_dir = f'{dest_mother_dir}/{object}'
            check_dir(dest_dir)
            
            num_img = int(num_per_cat*proportions[i]/100)
            print(f'{num_img} will be added from {src_dir}.\n')
            rand_idx = random.sample(range(0,num_src_img), num_img)
            for rand_idxx in rand_idx:
                from_image = f'{src_dir}/{src_dir_files[rand_idxx]}'
                shutil.copy(from_image, dest_dir)
                #print(f'{from_image} is copied to {dest_dir}')
            
        check_total_num_per_cat(dest_dir, num_per_cat, f'../../ProjectAlpha/Training_data/SNP/SNP_0.2/{purpose}/{object}')



if __name__ == '__main__':
    object_names = ['goldfinch','hamster','pitcher','upright','vizsla']
    purposes = ['val','train']
    training_set_names = ['333333', '204040', '404020']
    proportions = [[33,33,33],[20,40,40],[40,40,20]]
    num_per_cats = [130, 1040]

    for i in range(2):
        purpose = purposes[i]
        num_per_cat = num_per_cats[i]
        src_mother_dirs = list(f'../../ProjectAlpha/Training_data/SNP/{level}/{purpose}' for level in ['SNP_0.1', 'SNP_0.2', 'SNP_0.3'])
        for j in range(3):
            training_set_name = training_set_names[j]
            proportion = proportions[j]
            dest_mother_dir = f'./Training_data_mix/{training_set_name}/{purpose}'
            Mix_training_set_gen(src_mother_dirs, dest_mother_dir, object_names, purpose, training_set_name, num_per_cat, proportion)