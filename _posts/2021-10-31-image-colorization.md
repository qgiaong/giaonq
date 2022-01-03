---
excerpt: turn grayscale images into colorful masterpieces 
author_profile: true
title:  "Image Colorization"
categories:
  - machine-learning
tags:
  - deep learning
  - machine learning
  - computer vision
header:
  overlay_image: /assets/images/colors.jfif
  teaser: /assets/images/colors.jfif
  overlay_filter: 0.5
---
# Introduction
In this blog spot, we are going to tackle a Computer Vision problem called Image Colorization. In general, image colorization refers to converting a grayscale image (1 channel) to a full color image (3 channel). This is a challenging process because colorization is multi-modal, in such a way that a grayscale image can be mapped to several plausible colored images. 


Traditionally, this problem must be done manually and requires enormous attempt of human hardcoding. Instead, deep learning can be used to make this process automatic. We are going to make use of the ability of deep learning models to cature semantic information in images and to build a deep learning model using Pytorch. 

## Background on Colors
RGB format is often used to represent color images due to its simplicity. Each channel of this format represents a single color value, indicating how much Red, Green, Blue a pixel is. The color of the pixel by computing the sum of these 3 channels. The following plot gives an example of these 3 channels. The first image is the red channel, and we see that the red part of the original image is much darker in this channel.

![rgb](https://user-images.githubusercontent.com/43914109/147781992-f0b13a13-4de1-4fe8-ba69-be4a7243b20b.PNG)

Another common color format is the L\*a\*b format. In the L\*a\*b space, we also have 3 values but with different meaning. The first channel, L, represents the lightness of the pixel and contains the image in black-and-white. The *a  and *b values encode how much green-red and yellow-blue each pixel is, respectively. These 3 channels are visualized in the following plot: 
![lab](https://user-images.githubusercontent.com/43914109/147781998-6976ccd4-b091-46eb-966e-3210f668a499.png)

It is common for image colorization task to use L\*a\*b instead of RGB format, since we can separate the grayscale part (the lightness channel) directly. Therefore, we can generate the input data for the model directly and can formulate our problem as to reconstruct the  *a  and *b channel from the L channel. 

## Training strategy
As mentioned above, the task is to reconstruct the full-colored image using only the black-and-white image from the L channel. For this purpose, we adopt the pix2pix model provided by the [_**Image-to-Image Translation with Conditional Adversarial Networks**_](https://arxiv.org/abs/1611.07004) paper. 
The original implementation is provided in their [Github](https://github.com/cathmer/pix2pix/).

# Dataset

For training, we create a L\*a\*b dataset using existing images and creating grayscale versions of photos that models must learn to colorize. We make use of the 
[Lanscape Pictures Dataset](https://www.kaggle.com/arnaud58/landscape-pictures) containing  more than 4000 images of real-world landscape scenes. We make the problem a little bit easier, because the multimodality in nature scences is not so strong: sky is often blue and fields are green. This is not always the case: a car in grayscale image can have a red or blue color in real life. First, you need to download and unzip the dataset from Kaggle. We assume for the moment that the data is stored in the `data` folder.

Let's collect all the image paths:
```python
SEED = 42
BATCH_SIZE = 32
N_WORKERS = 2
HEIGHT = 256
WIDTH = 256
```
```python
path = "/content/data"
paths = np.array(glob.glob(path + "/*.jpg")) 
np.random.seed(SEED)
len(paths)
```
and divide the available data in training and validation set:
```python
rand_idxs = np.random.permutation(4319).astype(int)
train_idxs = rand_idxs[:3000] 
val_idxs = rand_idxs[3000:]
train_paths = paths[train_idxs]
val_paths = paths[val_idxs]
print(len(train_paths), len(val_paths))
```
```python
# plot some images
_, axes = plt.subplots(4, 4, figsize=(20, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    img = np.array(Image.open(img_path))
    ax.imshow(img)
    ax.axis("off")
```
## Define the DataLoader
Next, we should define the data loader that prepare the data for each training epoch. First of all, we must define the transformation function to convert images. We also do some data augmentation for the training set. Besides, scikit-learn provides convenient functions to switch between RGB and L\*a\*b spaces. This class below defines a custom dataset for our task:
```python
class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((HEIGHT, WIDTH)),
                transforms.RandomHorizontalFlip(), 
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((HEIGHT, WIDTH))
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. 
        ab = img_lab[[1, 2], ...] / 110. 
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)
```
Then we can create a custom data loader. This data loader shuffle the dataset in each iteration:
```python        
def make_dataloaders(path, split, batch_size=BATCH_SIZE, n_workers=N_WORKERS, pin_memory=True): 
    dataset = ColorizationDataset(path, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory, shuffle=True)
    return dataloader
    
train_dl = make_dataloaders(train_paths, 'train')
val_dl   = make_dataloaders(val_paths, 'val')
 ```  
 # Model Definition

## Introduction
First of all, recall that a GAN architecture is comprised of a generator that tries to generate new plausible images from some random value, while the discriminator attempts to classify images as fake (generated by the generator), or real image data. The two components are trained simultaneously in an adversarial process where the generator tries to fool the discriminator, while the discriminator attempts to classify as accurately as possible.


Put simply, a pix2pix networks extends the traditional GAN by some kind of input data. This is useful for the general purpose image-to-image translation. The discriminator is provided with both input and the target (or generated) image and must decide whether the target is a plausible transformation of the source image. While GAN attempts to learn a function mapping some random variable ![image](https://user-images.githubusercontent.com/43914109/147943830-901be011-629e-487a-8d9a-7f49f5aefcb9.png)
 to output image ![image](https://user-images.githubusercontent.com/43914109/147943876-cff0c533-76cc-41e2-8600-f56db4925fb4.png)
: ![image](https://user-images.githubusercontent.com/43914109/147943923-63d89967-0bec-4b43-a8be-066489550ea9.png)
, a pix2pix searches a a function that maps an input image ![image](https://user-images.githubusercontent.com/43914109/147943990-14bf9255-bae1-4991-93fb-7594097ce3dc.png), together with some random variable ![image](https://user-images.githubusercontent.com/43914109/147943830-901be011-629e-487a-8d9a-7f49f5aefcb9.png) to output image ![image](https://user-images.githubusercontent.com/43914109/147943876-cff0c533-76cc-41e2-8600-f56db4925fb4.png): ![image](https://user-images.githubusercontent.com/43914109/147944047-dd893530-cfda-4b13-9254-0692a829a475.png)



## The U-Net generator

The generator is an encoder-decoder architecture that employs an U-Net structure. It takes some input image (e.g. the grayscale one) and generates a target image (e.g. the colorized one). This is done by first mapping the input image on to a lower-dimensional bottleneck space, then upsampling the lower-dimensional representation to the size of the output image. The U-Net extends the traditional encoder-decoder architecture with skip-connections between the corresponding encoding-decoding layer. This forms the U-shape between the encoder-decoder components, hence the name.

![image](https://user-images.githubusercontent.com/43914109/147782892-6f57235b-a7b7-4b22-9ac7-af00522c87bf.png)

*Architecture of the U-Net Generator Model. Image extracted from the paper*
 
```python
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nf, inner_nf, submodule=None, input_nc=None, use_dropout=False,
                 innermost=False, outermost=False,  norm_layer=nn.BatchNorm2d):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost  # whether this is the outermost block 
        if input_nc is None: 
          input_nc = outer_nf # the number of channels in input images
        
        downconv = nn.Conv2d(input_nc, inner_nf, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nf)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nf)
        
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nf * 2, outer_nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nf, outer_nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nf * 2, outer_nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

```
```python
class Unet(nn.Module):
    def __init__(self, input_nc=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetSkipConnectionBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetSkipConnectionBlock(num_filters * 8, num_filters * 8, submodule=unet_block, use_dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetSkipConnectionBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetSkipConnectionBlock(output_c, out_filters, input_nc=input_nc, submodule=unet_block, outermost=True)
    
    def forward(self, x):
        return self.model(x)
```python
## The Markovian Discriminator (PatchGAN)
The discriminator does the conditional image classification, that is, it takes both the grayscale input image and the colored one, then outputs how likely that this is the real image or the image generated by the generator. The pix2pix discriminator views the image as a Markov random field, assuming independence between pixels separated by more than some distance. To this extend, the discriminator tries to classify if each $N × N$ patch in an image is real or fake (hence the name PatchGAN). Background on this is that the discriminator only focuses on the high-frequencies and relies on the L1 loss term of the generator for the correctness of the low frequencies.

The function below follows the paper and implements the $70×70$ PatchGAN discriminator. 

```python
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        k = 4
        s = 2
        p = 1
        model =  [nn.Conv2d(input_c, num_filters, k, s, p, bias=True)]  
        model += [nn.LeakyReLU(0.2, True)]
        s_ = 1 if i == (n_down-1) else 2
        model += [nn.Conv2d(num_filters * 2 ** i, num_filters * 2 ** (i + 1), k, s_, p, bias=False)]
        model += [nn.BatchNorm2d(num_filters * 2 ** (i + 1))]
        model += [nn.LeakyReLU(0.2, True)]
        model +=  [nn.Conv2d(num_filters * 2 ** n_down, 1, k, s, p, bias=True)] 
        self.model = nn.Sequential(*model)                                                   
        
    def forward(self, x):
        return self.model(x)
```
## The Loss function
**The generator loss**: The generator is trained via adversarial loss that encourages to generate plausible images in the target domain. The generator loss is a weighted sum of both the adversarial and L1 loss. The L1 loss term acts as a regularization term that attempts to minimize the absolute difference between the generated image and the target image. Why is it needed? This encorages the generator to generates plausible translation of the source image, instead of only plausible images of the target domain.  The author recommends to use a weighting of 100 to 1 in favor of the L1 loss. Another advantage of the L1 term is that it encourages less blurringin the generation.

<img src="https://latex.codecogs.com/svg.image?L_{L1}(G)&space;=&space;\mathbf{E}_{x,y,z}[\left\|y&space;-G(x,z)_1&space;\right\|]" title="L_{L1}(G) = \mathbf{E}_{x,y,z}[\left\|y -G(x,z)_1 \right\|]" />

<img src="https://latex.codecogs.com/svg.image?G*&space;=&space;arg\:&space;min_G\;&space;max_D&space;\;&space;L_{cGAN&space;}(G,&space;D)&space;&plus;&space;\lambda&space;L_{L1}(G)" title="G* = arg\: min_G\; max_D \; L_{cGAN }(G, D) + \lambda L_{L1}(G)" />

**The discriminator loss**: On the other hand, the discriminator employs a simple cross entropy loss to optimize its classification power.

## Model initialization
We initialize the model as it is suggested in the paper:
```python
def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model
```
Given all the components, we can define our colorization network as follows:

```python
class ColorizationModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def send_to_device(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)
        
    def forward(self):
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

```
Next, let's define the training loop and start training the model:
```python
losses = ['loss_D_fake', 'loss_D_real', 'loss_D',
          'loss_G_GAN', 'loss_G_L1', 'loss_G']
def compute_losses(model):
    loss_dict = {}
    for loss_name in losses:
        loss = getattr(model, loss_name)
        loss_dict.update({loss_name : loss.item()})
    return loss_dict

def train_model(model, train_dl, epochs, history, display_every=20):
    data = next(iter(val_dl)) 
    n_data = len(train_dl)
    for e in range(epochs):
        i = 0  
        loss_D_fake_sum, loss_D_real_sum, loss_D_sum,
          loss_G_GAN_sum, loss_G_L1_sum, loss_G_sum = 0                                
        for data in tqdm(train_dl):
            model.send_to_device(data) 
            model.optimize()
            loss_dict = compute_losses(model) 
            loss_D_fake, loss_D_real, loss_D, loss_G_GAN, loss_G_L1, loss_G = loss_dict.values()

            loss_D_fake_sum += loss_D_fake
            loss_D_real_sum+= loss_D_real
            loss_D_sum+= loss_D
            loss_G_GAN_sum+= loss_G_GAN
            loss_G_L1_sum+= loss_G_L1
            loss_G_sum+= loss_G

            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                visualize(model, data, save=False) 
        
        loss_D_fake_avg, loss_D_real_avg, loss_D_avg, loss_G_GAN_avg, loss_G_L1_avg, loss_G_avg  = \\
         loss_D_fake_sum/n_data, loss_D_real_sum/n_data, loss_D_sum/n_data,
          loss_G_GAN_sum/n_data, loss_G_L1_sum/n_data, loss_G_sum/n_data
        print(f"\nEpoch {e+1}/{epochs}, loss_D_fake_avg: {loss_D_fake_avg}, loss_D_real_avg: {loss_D_real_avg}, \\
         loss_D_avg: {loss_D_avg}, loss_G_GAN_avg: {loss_G_GAN_avg}, loss_G_L1_avg: {loss_G_L1_avg}, loss_G_avg: {loss_G_avg}")

```
```python
hist = []
model = ColorizationModel()
train_model(model, train_dl, 5, hist)
```
```python
# helper function to visualize data
def visualize(model, data, n_cols = 4, n_rows = 3):
    model.net_G.eval()
    with torch.no_grad():
        model.send_to_device(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = batch_lab_to_rgb(L, fake_color)
    real_imgs = batch_lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))

    for i in range(n_cols):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(n_rows, n_cols, i + 1 + n_cols)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(n_rows, n_cols, i + 1 + n_cols * 2)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()

```
Belows are some images colored by the model. The first row is the input, the last row contains the real images and the second row is the model output. We can see that our trained model does quite an impressive job and can generate plausible colorized, nice-looking images.
![result](https://user-images.githubusercontent.com/43914109/147783753-ca784391-f380-4313-8d6f-4b5a730cca7d.PNG)
