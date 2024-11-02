import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from basicsr.models import create_model  # NAFNet 모델 생성 코드
import argparse
from basicsr.utils.options import parse  # parse_options 임포트
import torch.multiprocessing
from torch.multiprocessing import freeze_support


class CustomDataset(Dataset):
    def __init__(self, clean_image_paths, noisy_image_paths, transform=None):
        self.clean_image_paths = [os.path.join(clean_image_paths, x) for x in os.listdir(clean_image_paths)]
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform
        self.center_crop = CenterCrop(224)
        self.resize = Resize((224, 224))

        self.noisy_clean_pairs = self._create_noisy_clean_pairs()

    def _create_noisy_clean_pairs(self):
        clean_to_noisy = {}
        for clean_path in self.clean_image_paths:
            clean_id = '_'.join(os.path.basename(clean_path).split('_')[:-1])
            clean_to_noisy[clean_id] = clean_path
        
        noisy_clean_pairs = []
        for noisy_path in self.noisy_image_paths:
            noisy_id = '_'.join(os.path.basename(noisy_path).split('_')[:-1])
            if noisy_id in clean_to_noisy:
                clean_path = clean_to_noisy[noisy_id]
                noisy_clean_pairs.append((noisy_path, clean_path))
        
        return noisy_clean_pairs

    def __len__(self):
        return len(self.noisy_clean_pairs)

    def __getitem__(self, index):
        noisy_image_path, clean_image_path = self.noisy_clean_pairs[index]
        noisy_image = Image.open(noisy_image_path).convert("RGB")
        clean_image = Image.open(clean_image_path).convert("RGB")
        
        noisy_image = self.center_crop(noisy_image)
        clean_image = self.center_crop(clean_image)
        noisy_image = self.resize(noisy_image)
        clean_image = self.resize(clean_image)
        
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

if __name__ == '__main__':
    
    # freeze_support()  # Windows에서 멀티프로세싱을 위해 필요
    
    # 시작 시간 기록
    start_time = time.time()

    # 데이터셋 경로
    noisy_image_paths = r'C:\ORIG\NAFNet\event\Training\noisy'
    clean_image_paths = r'C:\ORIG\NAFNet\event\Training\clean'
    train_transform = Compose([ToTensor()])
    train_dataset = CustomDataset(clean_image_paths, noisy_image_paths, transform=train_transform)

    batch_size = 8
    num_cores = os.cpu_count()
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=int(num_cores/2), shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 옵션 파일 경로 설정
    opt_path = r'C:\ORIG\NAFNet\options\train\GoPro\NAFNet-width64.yml'
    opt = parse(opt_path)  # parse 함수로 opt 생성

    # NAFNet 모델 초기화 및 GPU에 할당
    model = create_model(opt)  # NAFNet 모델 생성 코드
    # model.to(device)

    # 분산 학습 설정
    opt['dist'] = False  # 분산 학습을 사용하지 않도록 설정

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pretrained weight 불러오기
    pretrained_path = r'C:\ORIG\NAFNet\experiments\pretrained_models\NAFNet-GoPro-width64.pth'  # pretrained 모델의 경로

    if os.path.exists(pretrained_path):
        print("Pretrained model found. Loading weights...")
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # checkpoint에 'params' 키가 있는 경우 처리
        if 'params' in checkpoint:
            checkpoint = checkpoint['params']
        
        # state_dict의 키를 조정하여 "module." 접두사를 추가 또는 제거
        model_state_dict = model.net_g.state_dict()
        modified_checkpoint = {}
        
        for k, v in checkpoint.items():
            # "module." 접두사 제거
            if k.startswith("module."):
                k = k[7:]  # "module." 제거
            # 필요한 경우 "module." 접두사 추가
            if k not in model_state_dict and "module." + k in model_state_dict:
                k = "module." + k
            if k in model_state_dict:  # 현재 모델의 state_dict에 있는 키만 사용
                modified_checkpoint[k] = v

        # 누락된 키 확인
        missing_keys = set(model_state_dict.keys()) - set(modified_checkpoint.keys())
        if missing_keys:
            print("Warning: Missing keys in checkpoint:", missing_keys)

        # strict=False로 설정하여 일부 키가 일치하지 않아도 로드
        model.net_g.load_state_dict(modified_checkpoint, strict=False)
        print("Pretrained weights loaded successfully")
        
    # 손실 함수, 옵티마이저, 스케줄러 설정
    criterion = nn.L1Loss() # 무조건 바꿔야함!!!!!!!!!!!!
    # criterion =torch.nn.NLLLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    optimizer = optim.AdamW(model.net_g.parameters(), lr=0.0005, weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    # Resume state 로드
    resume_state = None
    state_folder_path = r'C:\ORIG\NAFNet\experiments\training_states'# 실험 폴더 경로
    try:
        states = os.listdir(state_folder_path)
        if states:
            max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
            resume_state = os.path.join(state_folder_path, max_state_file)
            print(f"Resuming from {resume_state}")
            checkpoint = torch.load(resume_state, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
        else:
            start_epoch = 0
            best_loss = float('inf')
    except FileNotFoundError:
        print("No previous states found. Starting fresh.")
        start_epoch = 0
        best_loss = float('inf')

    # 학습 루프
    num_epochs = 1
    for epoch in range(start_epoch, num_epochs):
        model.net_g.train()  # model.train() 대신
        running_loss = 0.0
        for noisy_images, clean_images in train_loader:
            noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
            optimizer.zero_grad()
            
            with autocast():
                # model(noisy_images) 대신 model.net_g(noisy_images) 사용
                outputs = model.net_g(noisy_images)
                loss = criterion(outputs, clean_images)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item() * noisy_images.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        # 모델 및 상태 저장
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # save_path = f'C:\ORIG\NAFNet\experiments\training_states\epoch_{epoch}.state'
            save_path = f'C:/ORIG/NAFNet/experiments/training_states/epoch_{epoch}.state'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss
            }, save_path)
            print(f'Best model saved at {save_path} with loss {best_loss:.4f}')

    # 총 소요 시간 계산
    end_time = time.time()
    training_time = end_time - start_time
    hours, minutes = divmod(training_time // 60, 60)
    seconds = int(training_time % 60)
    print(f"총 훈련 소요 시간: {int(hours)}시간 {int(minutes)}분 {seconds}초")
