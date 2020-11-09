from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import SlideDataset
from slideModel import Attention

from scipy.misc import imread
from tifffile import imsave
import openslide as opsl
import random
import scipy.io

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--patch_size', type=int, metavar='PS',
                    help='patch size')
parser.add_argument('--load_data', type=bool, default=True,
                    help='load all data first?')
parser.add_argument('--transform', type=bool, default=True,
                    help='map values?')                  
parser.add_argument('--slide_path',
                    help='which slide to process')
parser.add_argument('--mask_path',
                    help='where is associated mask')
parser.add_argument('--model_path', default='saved.model',
                    help='which model you want to use')
parser.add_argument('--bag_mean', type=int, default=5000,
                    help='how many tiles per bag (mean)')
parser.add_argument('--bag_std', type=int, default=1000,
                    help='tiles per bag std')
parser.add_argument('--num_query', type=int, default=0,
                    help='override default query points calculatation')
parser.add_argument('--query_type', default='tiled',
                    help='how to query tiles (tiled or random)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def generatePointsRandom():
    [width,height]=opsl.OpenSlide(args.slide_path).dimensions

    mask=imread(args.mask_path)
    [y,x]=np.nonzero(mask)
    minx=min(x)
    miny=min(y)
    maxx=max(x)
    maxy=max(y)
    print([minx,maxx,miny,maxy])

    # Figure out approximately how many query points cover the slide
    num=float(len(x))
    num=num*(width/mask.shape[1])*(height/mask.shape[0]) # appoximate area of mask pixels in full image
    num=int(num/(args.patch_size*args.patch_size))   # approximate number of patches with no overlap

    # Manual override for debugging
    if args.num_query:
        num=args.num_query

    # Generate bag sizes and query points
    nbags=num/args.bag_mean
    print('Generating ' + str(num) + ' query tiles over ' + str(nbags) + ' bags.')
    bag_points=[]
    bag_points_thumb=[]
    for n in range(0,nbags):
        bs=int(random.normalvariate(args.bag_mean,args.bag_std))
        i=0
        points=np.zeros((bs,2),dtype=np.int)
        points_thumb=np.zeros((bs,2),dtype=np.int)
        while i<bs:
            px=random.uniform(minx,maxx)
            py=random.uniform(miny,maxy)
            if mask[int(py),int(px)]>0: # remember that the mask is (H,W,D)
                points[i,0]=int(px*width/mask.shape[1])
                points[i,1]=int(py*height/mask.shape[0])
                points_thumb[i,0]=px
                points_thumb[i,1]=py
                i=i+1
        bag_points.append(points)
        bag_points_thumb.append(points_thumb)
    print('Done')
    return bag_points,bag_points_thumb

def generatePointsTiled():
    [width,height]=opsl.OpenSlide(args.slide_path).dimensions
    
    mask=imread(args.mask_path)
    [y,x]=np.nonzero(mask)
    minx=min(x)
    miny=min(y)
    maxx=max(x)
    maxy=max(y)
    print([minx,maxx,miny,maxy])
    
    # Compute interval in mask coordinates
    scale=width/mask.shape[1]
    mask_ps=float(args.patch_size)/float(scale)
    
    # Spaced query points
    xs=np.arange(minx,maxx,mask_ps)
    ys=np.arange(miny,maxy,mask_ps)
    points=[]
    points_thumb=[]
    num=0
    for i in range(0,len(ys)):
        for j in range(0,len(xs)):
            if mask[int(ys[i]),int(xs[j])]>0:
                points.append([int(xs[j]*scale),int(ys[i]*scale)])
                points_thumb.append([int(xs[j]),int(ys[i])])
                num=num+1
    random.shuffle(points)
    random.shuffle(points_thumb)
    points=np.array(points)
    points_thumb=np.array(points_thumb)
    nbags=int(num/args.bag_mean)
    print('Generating ' + str(num) + ' query tiles over ' + str(nbags) + ' bags.')
    bag_points=[]
    bag_points_thumb=[]
    bag_sizes=np.round(np.linspace(0,num,nbags)).astype(np.intp)
    for n in range(0,len(bag_sizes)-1):
        bag_points.append(points[bag_sizes[n]:bag_sizes[n+1]])
        bag_points_thumb.append(points_thumb[bag_sizes[n]:bag_sizes[n+1]])
    print('Done')
    return bag_points,bag_points_thumb

def makeBags(bag_points):
    loader = data_utils.DataLoader(SlideDataset(slide_path=args.slide_path,
                                          bag_points=bag_points,
                                          load_data=args.load_data,
                                          transform=args.transform,
                                          patch_size=args.patch_size),
                                 batch_size=1,
                                 shuffle=False,
                                **loader_kwargs)
    return loader
    
def attention_weights(loader):
    print('Getting attention weights')
    model.eval()
    weights=[]
    for batch_idx, (data) in enumerate(loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        loss, attention_weights = model.calculate_objective(data,Variable(torch.empty(1).cuda())) # Variable(0) is just a dummy
        attention_weights=attention_weights.data.cpu().numpy()
        weights.append(attention_weights)
        print('Bag '+str(batch_idx)+' done')
    print('All bags done')
    return weights

def makeMap(bag_points,bag_points_thumb,weights):
    [width,height]=opsl.OpenSlide(args.slide_path).dimensions
    map=np.zeros((height,width),dtype=np.float32) # H W D
    mask=imread(args.mask_path)
    map_thumb=np.zeros((mask.shape[0],mask.shape[1]),dtype=np.float32)
    for i in range(0,len(bag_points)):
        p=bag_points[i]
        p_thumb=bag_points_thumb[i]
        w=weights[i]
        #w=w.data.cpu().numpy()
        for j in range(0,p.shape[0]):
            map[p[j,1]:p[j,1]+args.patch_size,p[j,0]:p[j,0]+args.patch_size]=map[p[j,1]:p[j,1]+args.patch_size,p[j,0]:p[j,0]+args.patch_size]+w[0,j]
            map_thumb[p_thumb[j,1],p_thumb[j,0]]=map[p_thumb[j,1],p_thumb[j,0]]+w[0,j]
            
    print('Mapping done')
    map=map/np.max(map)
    map_thumb=map_thumb/np.max(map_thumb)
    print('Divide done')
    map=np.multiply(map,255)
    map_thumb=np.multiply(map_thumb,255)
    print('Multiply done')
    map=map.astype(dtype=np.uint8)
    map_thumb=map_thumb.astype(dtype=np.uint8)
    print('Casting done')
    return map,map_thumb
            
if __name__ == "__main__":
    print('loading model...')
    model=torch.load(args.model_path)
    if args.cuda:
        model.cuda()
    if args.query_type=="tiled":
        bag_points,bag_points_thumb=generatePointsTiled()
    elif args.query_type=="random":
        bag_points,bag_points_thumb=generatePointsRandom()
    loader=makeBags(bag_points)
    weights=attention_weights(loader)
    map,map_thumb=makeMap(bag_points,bag_points_thumb,weights)
    fn=args.mask_path.split("/")[-1]
    weights=np.concatenate(weights,axis=1)
    scipy.io.savemat('/isilon/datalake/cialab/scratch/cialab/tet/python/AttentionDeepMIL/heatmap/results_the_rest/'+fn+'.mat',dict(weights=weights))
    imsave('/isilon/datalake/cialab/scratch/cialab/tet/python/AttentionDeepMIL/heatmap/results_the_rest/'+fn+'.tif',map,compress=6)
    imsave('/isilon/datalake/cialab/scratch/cialab/tet/python/AttentionDeepMIL/heatmap/results_the_rest/'+fn+'_thumb.tif',map_thumb)
    
