python train_strong.py \
    --training-dataset pascal \
    --dataset-csv-path training_data/pascal-random/ \
    --batch-size 32 \
    --result-model-dir checkpoints/pascal/affine \
    --result-model-fn aug_resnet152_v2_pascal_random \
    --feature-extraction-cnn resnet152_v2 \
    --random-sample 1 \
    --num-epochs 50

# finetune with smaller learning rate
python train_strong.py \
    --training-dataset pascal \
    --dataset-csv-path training_data/pascal-random/ \
    --batch-size 32 \
    --result-model-dir checkpoints/pascal/affine \
    --result-model-fn aug_resnet152_v2_pascal_random_smallerlr_new \
    --feature-extraction-cnn resnet152_v2 \
    --random-sample 1 \
    --num-epochs 50 \
    --model checkpoints/pascal/affine/best_aug_resnet152_v2_pascal_random_strong_50_pascal_affine_resnet152_v2_grid_loss.pth.tar \
    --lr 0.0002
