import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import time
import shutil
from basicsr.utils.options import parse
from basicsr.utils.options import dict2str
from basicsr.models import create_model

class CustomDatasetTest(Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths) 
                                 if x.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = load_img(noisy_image_path)
        
        # Convert numpy array to PIL image
        if isinstance(noisy_image, np.ndarray):
            noisy_image = Image.fromarray(noisy_image)
        
        if self.transform:
            noisy_image = self.transform(noisy_image)
            
        return noisy_image, noisy_image_path

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 옵션 파일 경로 설정
    opt_path = '/content/unid_datathon_team3/options/train/GoPro/NAFNet-width64.yml'
    opt = parse(opt_path)
    opt['dist'] = False
    
    # NAFNet 모델 초기화
    model = create_model(opt)
    
    # Pretrained weight 불러오기
    pretrained_path = '/content/unid_datathon_team3/experiments/training_states/epoch_13.pth'
    
    if os.path.exists(pretrained_path):
        print("Pretrained model found. Loading weights...")
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        
        model_state_dict = model.net_g.state_dict()
        modified_checkpoint = {}
        
        for k, v in checkpoint.items():
            if k.startswith("module."):
                k = k[7:]
            if k not in model_state_dict and "module." + k in model_state_dict:
                k = "module." + k
            if k in model_state_dict:
                modified_checkpoint[k] = v

        missing_keys = set(model_state_dict.keys()) - set(modified_checkpoint.keys())
        if missing_keys:
            print("Warning: Missing keys in checkpoint:", missing_keys)

        model.net_g.load_state_dict(modified_checkpoint, strict=False)
        print("Pretrained weights loaded successfully")
    
    model.net_g.eval()
    model.net_g.to(device)
    
    # 데이터셋 경로
    test_input_dir = '/content/drive/MyDrive/Colab Notebooks/submission1'
    test_output_dir = '/content/drive/MyDrive/Colab Notebooks/Sample_Output5'
    
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    
    # Transform 설정
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 테스트 데이터셋 및 데이터로더 설정
    test_dataset = CustomDatasetTest(test_input_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Inference
    for noisy_image, noisy_image_path in test_loader:
        noisy_image = noisy_image.to(device)
        
        with torch.no_grad():
            denoised_image = model.net_g(noisy_image)
        
        # 후처리 및 저장
        denoised_image = denoised_image.cpu().squeeze(0)
        denoised_image = (denoised_image * 0.5 + 0.5).clamp(0, 1)
        denoised_image = transforms.ToPILImage()(denoised_image)
        
        # Save denoised image
        output_filename = os.path.basename(noisy_image_path[0])
        denoised_filename = os.path.join(test_output_dir, output_filename)
        denoised_image.save(denoised_filename)
        
        print(f'Saved denoised image: {denoised_filename}')

    # Create submission zip
    def zip_folder(folder_path, output_zip):
        shutil.make_archive(output_zip, 'zip', folder_path)
        print(f"Created {output_zip}.zip successfully.")
    
    zip_folder(test_output_dir, os.path.join(os.path.dirname(test_output_dir), 'submission'))

if __name__ == '__main__':
    start_time = time.time()
    test()
    end_time = time.time()
    test_time = end_time - start_time
    minutes, seconds = divmod(test_time, 60)
    print(f"Total test time: {int(minutes)}분 {int(seconds)}초")
