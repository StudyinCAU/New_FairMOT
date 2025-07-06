import cv2
import os
import glob

def images_to_video(img_dir):
    # 获取文件夹名作为视频名（val下一级目录名）
    base_name = os.path.basename(os.path.dirname(img_dir))
    output_path = f"/home/station1/weidongtang/report/work2/FairMOT/videos/{base_name}.mp4"

    # 获取所有jpg图像，按名称排序
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if not img_paths:
        print(f"未找到图像帧：{img_dir}")
        return

    # 读取第一张图像确定尺寸
    frame = cv2.imread(img_paths[0])
    height, width, _ = frame.shape

    # 定义视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 25, (width, height))

    for img_path in img_paths:
        frame = cv2.imread(img_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"视频已保存到：{output_path}")

# ✅ 修改此处，设置帧图像目录路径
img_dir = "/home/station1/weidongtang/report/work2/FairMOT/dataset/bdd100kmot_vehicle/images/val/b1c81faa-3df17267/img1"

# ✅ 调用主函数
images_to_video(img_dir)
