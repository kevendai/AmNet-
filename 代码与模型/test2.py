import torch
import cv2, datetime, os
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from os.path import join
from GCANet import AmNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def img2tensor(np_img):  # [h,w,c]
    tensor = get_transforms()(np_img).cuda()  # [c,h,w] [-1,1]
    tensor = tensor.unsqueeze(0)  # [b,c,h,w] [-1,1]
    return tensor


def get_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(),  # H,W,C -> C,H,W && [0,255] -> [0,1]
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) #[0,1] -> [-1,1]
    ])
    return transform


def tensor2img(one_tensor):  # [b,c,h,w] [-1,1]
    tensor = one_tensor.squeeze(0)  # [c,h,w] [0,1]
    tensor = tensor * 255  # [c,h,w] [0,255]
    tensor_cpu = tensor.cpu()
    img = np.array(tensor_cpu, dtype=np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img


if __name__ == "__main__":
    net = AmNet(in_c=3, out_c=3).to(device)
    with torch.no_grad():
        checkpoint_model = join('./Rlablemodel/',
                                '{}-model-epochs{}.pth'.format("AmNet", 190))
        checkpoint = torch.load(checkpoint_model, map_location='cpu')
        net.load_state_dict(checkpoint['model'])
        img_folder = "./examples/"
        pbar = tqdm(os.listdir(img_folder))
        for img_name in os.listdir(img_folder):
            img_path_raw = os.path.join(img_folder, img_name)
            img_raw = cv2.cvtColor(cv2.imread(img_path_raw), cv2.COLOR_BGR2RGB)
            im_w, im_h = img_raw.shape[1], img_raw.shape[0]
            img_raw = cv2.resize(img_raw, (im_w // 4 * 4, im_h // 4 * 4))
            img_raw_tensor = img2tensor(img_raw)
            output_tensor = net.forward(img_raw_tensor)
            output_img = tensor2img(output_tensor)
            save_folder = "./output/"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, img_name)
            # cv2.imwrite(save_path, output_img)
            cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            pbar.update(1)
