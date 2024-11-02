import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import img_as_ubyte
import cv2
from PIL import Image
from MPRNet import MPRNet
import utils

parser = argparse.ArgumentParser(description='Image Denoising using MPRNet')

parser.add_argument('--input_noisy_dir', default='C:/ORIG/NAFNet/event/Training/noisy', type=str, help='Directory of noisy images')
parser.add_argument('--input_clean_dir', default='C:/ORIG/NAFNet/event/Training/clean', type=str, help='Directory of clean images')
parser.add_argument('--result_dir', default='C:/ORIG/NAFNet/event/Test/output', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_denoising.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

# GPU 관련 설정 부분
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# GPU 사용 가능 여부 확인 및 device 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")  # GPU 모델명 출력
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU")

# GPU 메모리 상태 출력 (GPU 사용 가능한 경우)
if torch.cuda.is_available():
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    print(f"Available GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.0f}MB used")

# PSNR 계산 함수
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

# 이미지 로드 함수
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.
    return img

# 결과 디렉토리 생성
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

# 모델 초기화 및 설정 부분을 다음과 같이 수정
model_restoration = MPRNet()
model_restoration = model_restoration.to(device)

# weight 로드 부분 수정
try:
    checkpoint = torch.load(args.weights, map_location=device)
    if 'state_dict' in checkpoint:
        model_restoration.load_state_dict(checkpoint['state_dict'])
    else:
        model_restoration.load_state_dict(checkpoint)
    print("===>Testing using weights: ", args.weights)
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    print("Please check if the pretrained model is compatible")
    exit()

# GPU 사용 가능한 경우에만 DataParallel 적용
if torch.cuda.is_available():
    model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 이미지 파일 리스트 가져오기
noisy_files = sorted(os.listdir(args.input_noisy_dir))
clean_files = sorted(os.listdir(args.input_clean_dir))

# PSNR 값을 저장할 리스트
psnr_values = []

# 이미지 처리 - device 관련 부분 수정됨
with torch.no_grad():
    for idx, (noisy_file, clean_file) in enumerate(zip(noisy_files, clean_files)):
        # 파일 경로
        noisy_path = os.path.join(args.input_noisy_dir, noisy_file)
        clean_path = os.path.join(args.input_clean_dir, clean_file)
        
        # 이미지 로드
        noisy_img = load_img(noisy_path)
        clean_img = load_img(clean_path)
        
        # 텐서로 변환 - device 부분 수정됨
        noisy_tensor = torch.from_numpy(noisy_img).permute(2,0,1).unsqueeze(0).to(device)
        clean_tensor = torch.from_numpy(clean_img).permute(2,0,1).unsqueeze(0).to(device)
        
        # 이미지 복원
        restored = model_restoration(noisy_tensor)
        restored = torch.clamp(restored[0], 0, 1)
        
        # PSNR 계산
        psnr = calculate_psnr(restored, clean_tensor)
        psnr_values.append(psnr)
        
        # 결과 저장
        restored_img = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        
        # 이미지 저장
        noisy_save_path = os.path.join(args.result_dir, f'noisy_{noisy_file}')
        clean_save_path = os.path.join(args.result_dir, f'clean_{clean_file}')
        restored_save_path = os.path.join(args.result_dir, f'restored_{noisy_file}')
        
        cv2.imwrite(noisy_save_path, img_as_ubyte(noisy_img)[...,::-1])
        cv2.imwrite(clean_save_path, img_as_ubyte(clean_img)[...,::-1])
        cv2.imwrite(restored_save_path, img_as_ubyte(restored_img)[...,::-1])
        
        print(f'Image {idx+1}:')
        print(f'  Noisy image saved as: {noisy_save_path}')
        print(f'  Clean image saved as: {clean_save_path}')
        print(f'  Restored image saved as: {restored_save_path}')
        print(f'  PSNR = {psnr:.2f} dB')
        print('-' * 50)
        
        if idx >= 4:  # 처음 5개 이미지만 처리
            break

# 평균 PSNR 계산 및 출력
if psnr_values:
    average_psnr = sum(psnr_values) / len(psnr_values)
    print(f'\nAverage PSNR: {average_psnr:.2f} dB')
else:
    print("No images were processed")