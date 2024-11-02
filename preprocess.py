import os
import shutil
import time

def preprocess_data(base_dir):
    clean_dir = os.path.join(base_dir, 'clean')
    noisy_dir = os.path.join(base_dir, 'noisy')

    # Clean 폴더와 Noisy 폴더를 만듭니다.
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    # 각 조도 폴더 내의 GT 폴더와 나머지 폴더들을 분류합니다.
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            current_dir = os.path.join(root, dir_name)
            
            # 'GT'가 이름에 포함된 폴더는 'clean' 디렉토리로 이동
            if '원본_GT' in dir_name:
                for filename in os.listdir(current_dir):
                    if filename.endswith('.jpg'):
                        shutil.move(os.path.join(current_dir, filename), os.path.join(clean_dir, filename))
            
            # 그 외의 폴더는 'noisy' 디렉토리로 이동
            elif dir_name not in ['clean', 'noisy']:
                for filename in os.listdir(current_dir):
                    if filename.endswith('.jpg'):
                        shutil.move(os.path.join(current_dir, filename), os.path.join(noisy_dir, filename))

    # 불필요한 폴더 삭제
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_name not in ['clean', 'noisy']:
                for attempt in range(5):  # 최대 5번 시도
                    try:
                        shutil.rmtree(dir_path)
                        break  # 삭제에 성공하면 루프를 빠져나감
                    except PermissionError:
                        print(f"잠금 해제 대기 중: {dir_path}")
                        time.sleep(1)  # 잠금 해제 대기 후 재시도

    print(f'{base_dir} 데이터 전처리가 완료되었습니다.')

# 경로 설정
data_dir = r'C:\ORIG\NAFNet\event'
training_base_dir = os.path.join(data_dir, 'Training')
validation_base_dir = os.path.join(data_dir, 'Validation')

preprocess_data(training_base_dir)
preprocess_data(validation_base_dir)
