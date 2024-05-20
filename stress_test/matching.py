import sys
import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import lmdb
import time
import faiss

sys.path.append("./SOLAR")

from SOLAR.solar_global.utils.networks import load_network

def color_norm(im, mean, std):
    """Performs per-channel normalization (CHW format)."""
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im

def prepare_im(im):
    _MEAN = [0.485, 0.456, 0.406]
    _SD = [0.229, 0.224, 0.225]
    """Prepares the image for network input."""
    im = im.transpose([2, 0, 1])
    # [0, 255] -> [0, 1]
    im = im / 255.0
    # Color normalization
    im = color_norm(im, _MEAN, _SD)
    im = torch.from_numpy(im)
    im = im.unsqueeze(0)
    return im

def extract_feat(net, image):
    # _scale_list = [0.7071, 1, 1.4142]
    _scale_list = [1]


    img_ = image
    img_query = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    im = img_query
    im_ = cv2.resize(im, (256, int(im.shape[0] * 256 / im.shape[1])))
    im = im_.astype(np.float32, copy=False)
    im = prepare_im(im)

    v = torch.zeros(net.meta['outputdim'])
    
    for s in _scale_list:
        if s == 1:
            _input_t = im.clone()
        else:
            _input_t = nn.functional.interpolate(im, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(_input_t).pow(1).cpu().data.squeeze()
    v /= len(_scale_list)
    v = v.pow(1./1)
    v /= v.norm()
    return v.numpy().astype(np.float16)

class search_solar():
    def __init__(self, network_path='resnet101-solar-best.pth', device='cuda', lmdb_path=''):
        self.device = device
        net = load_network(network_path=network_path, device=device)
        net.eval()
        print(net.meta_repr())
        self.model = net

        if lmdb_path != '':
            self.features_DB, self.ids_list = self.load_features_from_lmdb(lmdb_path)
            print("self.features_DB: ", self.features_DB.shape)
            print("self.ids_list: ", self.ids_list)
            self.features_DB = self.features_DB.cpu().numpy()
            self.features_DB_tensor = torch.tensor(self.features_DB, device=self.device)
            self.features_DB_tensor /= self.features_DB_tensor.norm(dim=-1, keepdim=True)


    def gen_db(self, folder_data, lmdb_path):
        os.makedirs(lmdb_path, exist_ok=True)
        env = lmdb.open(lmdb_path, map_size=1e12)

        with env.begin(write=True) as txn:
            for path in tqdm(os.listdir(folder_data)):
                name = os.path.splitext(path)[0]
                img_path = os.path.join(folder_data, path)
                image = cv2.imread(img_path)
                feat_q = extract_feat(self.model, image)
                print("Saving feature with shape:", feat_q.shape, "and dtype:", feat_q.dtype)
                txn.put(name.encode(), feat_q.tobytes())

        env.close()
        return lmdb_path

    def load_features_from_lmdb(self, lmdb_path):
        env = lmdb.open(lmdb_path, readonly=True)
        features_list = []
        ids_list = []

        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                ids_list.append(key.decode())
                features_list.append(np.frombuffer(value, dtype=np.float16))
        
        features_array = np.stack(features_list)
        features_tensor = torch.tensor(features_array, dtype=torch.float16).to(self.device)
        
        return features_tensor, ids_list

    def search_image(self, image, top_k=5):
        t0 = time.time()
        feat_q = extract_feat(self.model, image)
        feat_q = torch.tensor(feat_q, device=self.device, dtype=torch.float16).unsqueeze(0)

        # Normalize features
        feat_q /= feat_q.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (feat_q @ self.features_DB_tensor.T).squeeze(0).cpu().numpy()

        # Filter and sort by similarity score
        top_indices = np.argsort(similarity)[-top_k:][::-1]
        results = [(self.ids_list[idx], similarity[idx]) for idx in top_indices]

        print("Search results:", results)
        print("Time per image:", time.time() - t0)
        return results


if __name__ == '__main__':
    solar_search = search_solar(network_path='./SOLAR/data/networks/resnet101-solar-best.pth', device='cuda', lmdb_path='./data/20240520_feature_DB')

    # Uncomment to generate the database
    # solar_search.gen_db(folder_data="./data/20240520_Data_product", lmdb_path="./data/20240520_feature_DB")

    folder_check = "./data/20240520_Data_product"
    
    for path in os.listdir(folder_check):
        image = cv2.imread(os.path.join(folder_check, path))
        print("image: ", image.shape, os.path.join(folder_check, path))
        result = solar_search.search_image(image=image, top_k=5)
