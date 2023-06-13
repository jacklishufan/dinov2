from matplotlib import pyplot as plt
import wandb
def wandb_dump_img(imgs,category,commit=False):
    if wandb.run is None:
        return
    n_imgs = len(imgs)
    fig, axes = plt.subplots(1,n_imgs,figsize=(5*n_imgs, 5))
    #raw, kmeans on 
    fig.tight_layout()
    for idx,img in enumerate(imgs):
        axes[idx].imshow(img)
    wandb.log({category:wandb.Image(fig)},commit=commit) 
    plt.close(fig)