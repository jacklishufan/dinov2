from dinov2.data.datasets import ImageNet
import argparse
parser = argparse.ArgumentParser()

ROOT = '/home/jacklishufan/imagenet/ILSVRC/in1k'
EXTRA = '/home/jacklishufan/imagenet/extra'

parser.add_argument('--root',default=ROOT,type=str)
parser.add_argument('--extra',default=EXTRA,type=str)
args = parser.parse_args()
for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=args.root, extra=args.extra)
    dataset.dump_extra()