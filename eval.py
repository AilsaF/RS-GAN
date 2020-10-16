import argparse
import inception_score_tf
import fid_tf
import prd_score as prd
import networks
import datasets
import numpy as np
import torch

def getRealData(size):
    loader = datasets.getDataLoader(args.dataset, args.image_size, batch_size=size, train=False)
    data_iter = iter(loader)
    realdata = data_iter.next()[0]
    realdata = np.array(realdata)
    if args.dataset == 'mnist':
        realdata = realdata.repeat(3, axis=1)
    realdata = (realdata / 2 + 0.5) * 255
    realdata = realdata.astype(np.uint8)
    return realdata

def getFakedata(size):
    data = []
    for _ in range(0, size, 500):
        z = torch.randn(500, args.input_dim).cuda()
        with torch.no_grad():
            x = netG(z).cpu().numpy()
            if args.dataset == 'mnist':
                x = x.repeat(3, axis=1)
        data.append(x)
    data = np.concatenate(data)
    data = (data / 2 + 0.5) * 255
    data = data.astype(np.uint8)
    return data

def getFID():
    realdata = getRealData(10000)
    fakedata = getFakedata(10000)
    fid = fid_tf.get_fid(realdata, fakedata)
    print("FID = ", fid)
    return fid

def getIS():
    data = getFakedata(50000)
    mean, std = inception_score_tf.get_inception_score(data, splits=10)
    print("IS = ", mean, std)
    return mean, std


def getPRD():
    realdata = getRealData(10000)
    fakedata = getFakedata(10000)
    ref_emb = fid_tf.get_inception_activations(realdata)
    eval_emb = fid_tf.get_inception_activations(fakedata)
    prd_res = prd.compute_prd_from_embedding(eval_data=eval_emb, ref_data=ref_emb)
    print("PRD =", prd_res)
    return prd_res


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='IS', choices=['IS', 'FID', 'PRD'])
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--structure', type=str, default='dcgan', choices=['resnet', 'dcgan'])
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--num_features', type=int, default=64)

    args = parser.parse_args()

    netG, _ = networks.getGD_SN(args.structure, args.dataset, args.image_size, args.num_features, dim_z=args.input_dim, ignoreD=True)
    netG.load_state_dict(torch.load(args.model_path))
    netG.cuda()
    print(args.model_path)
    if args.metric == 'IS':
        getIS()
    elif args.metric == 'FID':
        getFID()
    elif args.metric == 'PRD':
        getPRD()
