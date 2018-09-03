python train_strong.py \
    --training-dataset pascal \
    --dataset-csv-path training_data/pascal-random/ \
    --batch-size 32 \
    --result-model-dir checkpoints/pascal/tps \
    --result-model-fn aug_resnet_pascal_random \
    --feature-extraction-cnn resnet101 \
    --random-sample 1 \
    --num-epochs 50 \
    --geometric-model tps

# finetune with smaller learning rate
python train_strong.py \
    --training-dataset pascal \
    --dataset-csv-path training_data/pascal-random/ \
    --batch-size 32 \
    --result-model-dir checkoints/pascal/tps \
    --result-model-fn aug_resnet_pascal_random_smallerlr_new \
    --feature-extraction-cnn resnet101 \
    --random-sample 1 \
    --num-epochs 50 \
    --model checkpoints/pascal/tps/best_aug_resnet_pascal_random_strong_50_pascal_tps_resnet101_grid_loss.pth.tar \
    --lr 0.0002 \
    --geometric-model tps
