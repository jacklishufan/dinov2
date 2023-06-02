CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    dinov2/train/train.py \
    --config-file dinov2/configs/train/vit_s14_short.yaml \
    --output-dir output_dir3 \
    train.dataset_path=ImageNet:split=TRAIN:root=/home/jacklishufan/imagenet/ILSVRC/in1k:extra=/home/jacklishufan/imagenet/extra


torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    dinov2/eval/knn.py \
    --config-file dinov2/configs/train/vit_s14_short.yaml \
    --pretrained-weights output_dir2/eval/training_124999/teacher_checkpoint.pth \
    --output-dir  output_dir2/eval/training_124999/knn \
    --train-dataset ImageNet:split=TRAIN:root=/home/jacklishufan/imagenet/ILSVRC/in1k:extra=/home/jacklishufan/imagenet/extra \
    --val-dataset ImageNet:split=VAL:root=/home/jacklishufan/imagenet/ILSVRC/in1k:extra=/home/jacklishufan/imagenet/extra