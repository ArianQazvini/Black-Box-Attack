import torch
from torch.backends import cudnn
import pickle
from PIL import Image
import torchvision.transforms.functional as TF
import json
import random
from torchvision import transforms
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
import gc

def memory_info(device_id):
    device_name = torch.cuda.get_device_name(device_id)
    total_memory = torch.cuda.get_device_properties(1).total_memory
    memory_allocated = torch.cuda.memory_allocated(1)
    memory_cached = torch.cuda.memory_reserved(1)
    
    print(f"Device {device_id}: {device_name}")
    print(f"  Total Memory: {total_memory / (1024**3):.2f} GB")
    print(f"  Memory Allocated: {memory_allocated / (1024**2):.2f} MB")
    print(f"  Memory Cached: {memory_cached / (1024**2):.2f} MB")

class RIVAL10(Dataset):    
    def __init__(self, train=True, return_masks=False, data_root="/datasets/RIVAL10"):

        self.train = train
        self.return_masks = return_masks
        self.instance_types = ['ordinary']

        root_data = data_root + "/{}/"
        self.data_root = root_data.format('train' if self.train else 'test')

        root_mask = data_root + "/{}/entire_object_masks/"
        self.mask_root = root_mask.format('train' if self.train else 'test')

        with open(data_root + "/meta/label_mappings.json", 'r') as f:
            self.label_mappings = json.load(f)
        with open(data_root + "/meta/wnid_to_class.json", 'r') as f:
            self.wnid_to_class = json.load(f)

        self.collect_instances()
        self.collect_images()

    def get_rival10_og_class(self, img_url):
        wnid = img_url.replace('\\', '/').split('/')[-1].split('_')[0]
        inet_class_name = self.wnid_to_class[wnid]
        classname, class_label = self.label_mappings[inet_class_name]
        return classname, class_label

    def collect_instances(self):
        self.instances_by_type = dict()
        self.all_instances = []
        for subdir in self.instance_types:
            instances = []
            dir_path = self.data_root + subdir
            for f in tqdm(glob.glob(dir_path + '/*')):
                if '.JPEG' in f and 'merged_mask' not in f:
                    img_url = f
                    label_path = f[:-5] + '_attr_labels.npy'
                    merged_mask_path = f[:-5] + '_merged_mask.JPEG'
                    mask_dict_path = f[:-5] + '_attr_dict.pkl'
                    instances.append((img_url, label_path, merged_mask_path, mask_dict_path))
            self.instances_by_type[subdir] = instances.copy()
            self.all_instances.extend(self.instances_by_type[subdir])

    def transform(self, imgs):
        transformed_imgs = []
        resize = transforms.Resize((224, 224))
        i, j, h, w = transforms.RandomResizedCrop.get_params(imgs[0], scale=(0.8, 1.0), ratio=(0.75, 1.25))
        coin_flip = (random.random() < 0.5)
        for ind, img in enumerate(imgs):
            if self.train:
                img = TF.crop(img, i, j, h, w)

                if coin_flip:
                    img = TF.hflip(img)

            img = TF.to_tensor(resize(img))

            if img.shape[0] == 1:
                img = torch.cat([img, img, img], axis=0)

            transformed_imgs.append(img)

        return transformed_imgs

    def __len__(self):
        return len(self.all_instances)

    def collect_images(self):

        self.all_images = []

        for img_url, label_path, merged_mask_path, mask_dict_path in tqdm(self.all_instances):

            class_name, class_label = self.get_rival10_og_class(img_url)

            img = Image.open(img_url)
            if img.mode == 'L':
                img = img.convert("RGB")

            imgs = [img]

            if self.return_masks:
                merged_mask_img = Image.open(merged_mask_path)
                imgs = [img, merged_mask_img]

            imgs = self.transform(imgs)

            if self.return_masks:
                self.all_images.append([imgs[0], imgs[1], class_label])
            else:
                self.all_images.append([imgs[0], class_label, img_url])

    def __getitem__(self, i):

        return self.all_images[i]

class Args:
    def __init__(self, epsilon, niters, batch_size, save_dir, target_attack, s_num, r_flag):
        self.epsilon = epsilon
        self.niters = niters
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.target_attack = target_attack
        self.s_num = s_num
        self.r_flag = r_flag

def load_dataset(num_samples=300,batch_size=64, rival_mask=False):
    
        trainset = RIVAL10(train=True, return_masks=rival_mask)
        subset_indices = list(range(num_samples))
        rival_subset = Subset(trainset, subset_indices)
        trainloader = torch.utils.data.DataLoader(rival_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        num_classes = 10


        return trainloader, num_classes

def compute_ig(inputs, label_inputs, model,steps):
    baseline = np.zeros(inputs.shape)
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    scaled_inputs = np.asarray(scaled_inputs)

    if args.r_flag:
        # This is an approximate calculation of TAIG-R
        scaled_inputs += np.random.uniform(-args.epsilon, args.epsilon, scaled_inputs.shape)

    scaled_inputs = torch.from_numpy(scaled_inputs)
    scaled_inputs = scaled_inputs.to(device, dtype=torch.float)
    scaled_inputs.requires_grad_(True)
    att_out = model(scaled_inputs)
    score = att_out[:, label_inputs]
    loss = -torch.mean(score)
    model.zero_grad()
    loss.backward()
    grads = scaled_inputs.grad.data
    avg_grads = torch.mean(grads, dim=0)
    delta_X = scaled_inputs[-1] - scaled_inputs[0]
    integrated_grad = delta_X * avg_grads
    IG = integrated_grad.cpu().detach().numpy()

    del integrated_grad, delta_X, avg_grads, grads, loss, score, att_out
    return IG

def create_adv_imgs(ori_loader,num_classes,device, args):
    
    epsilon = args.epsilon / 255
    batch_size = args.batch_size
    save_dir = args.save_dir
    niters = args.niters
    target_attack = args.target_attack
    r_flag = args.r_flag
    s_num = int(args.s_num)
    
    #surrogate model resnet50
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    
    

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    input_size = [3, 224, 224]
    norm = T.Normalize(mean=mean, std=std)
    resize = T.Resize((input_size[1], input_size[2]))
    model = nn.Sequential(
        resize,
        norm,
        model
    )
    model.eval()

    if target_attack:
        label_switch = torch.tensor(list(range(500, 1000)) + list(range(0, 500))).long()

    true_labels = []
    adv_imgs = []
    original_imgs = []

    for ind, ori_img in enumerate(ori_loader):
        label = ori_img[1]
        true_labels.append(label)

        if target_attack:
            label = label_switch[label]

        ori_img[0] = ori_img[0].to(device)
        original_imgs.append((ori_img[0].data))
        img = ori_img[0].clone()

        for i in tqdm(range(niters), desc="Iterations Progress"):
            img_x = img
            img_x.requires_grad_(True)
            steps = s_num
            igs = []

            for im_i in range(img_x.shape[0]):
                inputs = img_x[im_i].cpu().detach().numpy()
                label_inputs = label[im_i]
                integrated_grad = compute_ig(inputs, label_inputs, model,steps)
                igs.append(integrated_grad)

            igs = np.array(igs)
            model.zero_grad()
            input_grad = torch.from_numpy(igs).to(device)

            if target_attack:
                input_grad = -input_grad

            img = img.data + 1. / 255 * torch.sign(input_grad)
            img = torch.where(img > ori_img[0] + epsilon, ori_img[0] + epsilon, img)
            img = torch.where(img < ori_img[0] - epsilon, ori_img[0] - epsilon, img)
            img = torch.clamp(img, min=0, max=1)

        adv_imgs.append(img.data)
        # np.save(save_dir + '/batch_{}.npy'.format(ind), torch.round(img.data * 255).cpu().numpy().astype(np.uint8()))
        # del img, ori_img, input_grad
        # print('batch_{}.npy saved'.format(ind))


    true_labels = torch.cat(true_labels)
    
    return true_labels,original_imgs,adv_imgs


def compute_asr(batch_size, true_labels, original_imgs, adv_imgs):
    
    model_res18 = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model_res18.fc.in_features
    model_res18.fc = nn.Linear(num_ftrs, 10)
    

    
    model_path = "/models/res18-ce/model_LastEpoch.pth"
    
    
    # #------------------------

    memory_info(1)
    model_res18.load_state_dict(torch.load(p))
    model_res18 = model_res18.to(device)
    model_res18.eval()
        
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, images in enumerate(original_imgs):

            labels = true_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_res18(images)
            _, org_predicted = torch.max(outputs, 1)
                
  
            total += labels.size(0)
            correct += (org_predicted == labels).sum().item()
        
      accuracy = 100 * correct / total
      info = p.split('/')
      print(f'Accuracy for {info[-1]}-{info[-2]}: {accuracy:.2f}%')
        

        
    mismatch= 0
    total = 0
    with torch.no_grad():
        for batch_idx, adv_images in enumerate(adv_imgs):
                
            labels = true_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            adv_images = adv_images.to(device)
            labels = labels.to(device)
                
            outputs = model_res18(adv_images)
            _, adv_predicted = torch.max(outputs, 1)

            total += labels.size(0)
            mismatch += (adv_predicted != labels).sum().item()
         
    asr = 100 * mismatch / total
    print(f'ASR for {info[-1]}-{info[-2]}: {asr:.2f}%')
    print("--------------")

def save_imgs(true_labels,original_imgs,adv_imgs, tl_path, adv_path, org_path):
    

    with open(tl_path, 'wb') as file:
        pickle.dump(true_labels, file)

    with open(adv_path, 'wb') as file:
        pickle.dump(adv_imgs, file)

    with open(org_path, 'wb') as file:
        pickle.dump(original_imgs, file)
        
def load_imgs(tl_path, adv_path, org_path):
    
    true_labels = []
    adv_imgs = []
    original_imgs = []
    with open(tl_path,'rb') as file:
        true_labels = pickle.load(file)
    
    with open(adv_path,'rb') as file:
        adv_imgs = pickle.load(file)

    with open(org_path,'rb') as file:
        original_imgs = pickle.load(file)
    
    return true_labels,original_imgs,adv_imgs
        
if __name__ == "__main__":
    cuda_available = torch.cuda.is_available()
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:

        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA Devices: {num_devices}")
        

        # memory_info(1)
    
    torch.cuda.empty_cache()        
    device = torch.device("cuda:1")
    print('device: ',device)
    
    args = Args(
    epsilon=128,
    niters=20,
    batch_size=128,
    save_dir='test/',
    target_attack=False,
    s_num='20',
    r_flag=False,
    )


    #-------------------------


    # file_path = '/home/aut140302_ria/workspace/RIA/datasets/RIVAL10/rival10_train_arian.pkl'  #batch size = 5 
    # ori_loader = torch.load(file_path)
    # num_classes=10 
    
    ori_loader, num_classes=  load_dataset(num_samples=1000,batch_size=args.batch_size)
    
    #-----------------------------
    
    print('here')
    true_labels,original_imgs,adv_imgs = create_adv_imgs(ori_loader=ori_loader,num_classes=num_classes,device=device,args=args)
    #------------
    tl_path = '/data/true_labels_1000_20itr_samples.pkl'
    adv_path= '/data/adv_images_1000_20itr_samples.pkl'
    org_path = '/data/original_images_1000_20itr_samples.pkl'

    save_imgs( true_labels, original_imgs, adv_imgs, tl_path, adv_path, org_path)
    
    # true_labels,original_imgs,adv_imgs= load_imgs(tl_path, adv_path, org_path)


    compute_asr(args.batch_size, true_labels, original_imgs, adv_imgs)


    
    

    
    
    

    

        
      
