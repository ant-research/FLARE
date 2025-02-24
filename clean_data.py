import os

# 指定图片文件夹路径
folder_path = '/nas3/zsz/FLARE_clean/assets/gt_cam/cams'

# 定义保留的编号
keep_numbers = list(range(1, 200, 32))

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件名是否符合格式，并提取编号
    if filename.endswith('.txt'):
        try:
            number = int(filename.split('.')[0].split('_')[0])  # 假设文件名是编号.jpg的形式
            if number not in keep_numbers:
                # 如果编号不在保留列表中，则删除文件
                os.remove(os.path.join(folder_path, filename))
                print(f"Deleted file: {filename}")
            else:
                os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, f'{number:05d}.txt'))
        except ValueError:
            # 如果文件名不符合编号.jpg的格式，跳过
            print(f"Skipping file: {filename} (invalid format)")
            

# 指定图片文件夹路径
folder_path = '/nas3/zsz/FLARE_clean/assets/gt_cam/images'

# 定义保留的编号

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件名是否符合格式，并提取编号
    if filename.endswith('.png'):
        try:
            number = int(filename.split('.')[0].split('_')[-1])  # 假设文件名是编号.jpg的形式
            if number not in keep_numbers:
                # 如果编号不在保留列表中，则删除文件
                os.remove(os.path.join(folder_path, filename))
                print(f"Deleted file: {filename}")
            else:
                os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, f'{number:05d}.png'))
        except ValueError:
            # 如果文件名不符合编号.jpg的格式，跳过
            print(f"Skipping file: {filename} (invalid format)")