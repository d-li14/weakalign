#cd ..
python train_weak.py --feature-extraction-cnn resnet101 --model-aff trained_models/best_aug_resnet_pascal_random_smallerlr_new_strong_50_pascal_affine_resnet101_grid_loss.pth.tar --model-tps trained_models/best_aug_resnet_pascal_random_smallerlr_new_strong_50_pascal_tps_resnet101_grid_loss.pth.tar  --training-dataset pf-pascal --dataset-csv-path training_data/pf-pascal-flip/ --dataset-image-path datasets/proposal-flow-pascal/ --num-epochs 15 --lr 5e-8 --result-model-fn train_weak
