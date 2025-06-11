import os
import torch
torch.cuda.set_device(2)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import shutil
from attack import *
from tools import *
from dataload import *

def run_attack_and_evaluate(attack, model_names, input_dir, output_dir, device, batch_size=64, img_size=224):
    print("[1] training --------------------------------------------")
    imagenet_dataset = AdvDataset(root_dir=input_dir, mode='train', img_size=img_size)
    print("[2] loading--------------------------------------------")
    train_loader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("[3] generating--------------------------------------------")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for images, labels, filenames in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        delta = attack(images, labels) 
        adv_images = torch.clamp(images + delta, 0, 1)  
        save_images(adv_images, filenames, output_dir) 
    
    
    print("[4] receiving--------------------------------------------")
    eval_dataset = AdvDataset(root_dir=output_dir, mode='eval', img_size=img_size)
    print("[5] loading--------------------------------------------")
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    results = {}
    print(f"[6] attack {model_names}")
    
    for name in model_names:
        model = load_model(name, device).eval().to(device)
        correct, total = 0, 0
        for images, labels, filenames in eval_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        asr = 100 * (1 - correct / total) 
        results[name] = asr
        print(f"{name} ASR: {asr:.1f}%")
    
    return results


def main():
    # 定义常量参数
 
    input_dir = ''
    output_dir = ''
    source_model = 'resnet18'
    model_list = ['resnet18', 'swin_b', 'inception_v3', 'resnet101', 'vit_b_16','vit_l_16',
                  'vit_small_r26_s32_224','mobilevit_s', 'twins_pcpvt_base']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epsilon = 16 / 255
    alpha = 1.6 / 255
    epochs = 10
    num_scale = 5
    num_block = 3
    num_global = 15
    batch_size = 48
    img_size = 224
    random_dataset = AdvDataset(root_dir=your_randx_path, mode='train', img_size=img_size)
    
    attack = DSA(model_name=source_model, 
                 epsilon=epsilon, 
                 alpha=alpha, 
                 epoch=epochs, 
                 num_scale=num_scale, 
                 num_block=num_block, 
                 device=device, 
                 num_global=num_global, 
                 dataset=random_dataset
                 )
    
    print("Running DSA attack......")
    run_attack_and_evaluate(attack, model_list, input_dir, output_dir, device=device,batch_size=batch_size, img_size=img_size)

img_min = 0.0
img_max = 1.0
print(f"GPU counts: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.current_device()}")

if __name__ == '__main__':
    main()

