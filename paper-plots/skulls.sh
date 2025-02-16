FILE=skull_variance.csv

read -p "Train new model? [y/N] " TRAIN_NEW

if [ "$TRAIN_NEW" = "y" ]; then
    python thesis/scripts/common.py --rng_key 42 --model LongVariance --network InverseUNetVarianceEmbedding --max_hidden_size 2048 --activation relu --objective Novel --constraints SkullLandmarks --landmarks_info ../data/canidae/skull_landmarks_information.csv --initial_skull ../data/canidae/landmarks/al_Canislupus_Bergen_B2.csv --terminal_skull ../data/canidae/landmarks/al_Vulpes_vulpes-000371189.csv --every 1 --bone 9 --diffusion KunitaLong --variance 1 --gamma 0.001 --min_diffusion_scale -2 --max_diffusion_scale 0 --simulator LongSimulator --displacement True --n 100 --epochs 5000 --learning_rate 1e-3

    read -p "Enter version number: " VERSION
else
    read -p "Enter existing version number: " VERSION
fi

echo "species,method,sigma,ll" > $FILE

python thesis/inference/variance.py --rng_key 42 --constraints SkullLandmarks --landmarks_info ../data/canidae/skull_landmarks_information.csv --initial_skull ../data/canidae/landmarks/al_Canislupus_Bergen_B2.csv --terminal_skull ../data/canidae/landmarks/al_Canislupus_Bergen2698.csv --every 1 --bone 9 --diffusion BrownianWideKernel --variance 1 --gamma 0.001 --simulator AutoLongSimulator --n_mc 100 --n_steps 1000  --from_sigma -2 --to_sigma 0 --n_values 50 --model LongVariance --network InverseUNetVarianceEmbedding --max_hidden_size 2048 --activation relu --objective Novel --displacement True --checkpoint logs/SkullLandmarks_KunitaLong/version_${VERSION}/checkpoints/  | sed -e "s/^/Canis lupus,/" >> $FILE
python thesis/inference/variance.py --rng_key 42 --constraints SkullLandmarks --landmarks_info ../data/canidae/skull_landmarks_information.csv --initial_skull ../data/canidae/landmarks/al_Canislupus_Bergen_B2.csv --terminal_skull ../data/canidae/landmarks/al_Vulpes_vulpes-000371189.csv --every 1 --bone 9 --diffusion BrownianWideKernel --variance 1 --gamma 0.001 --simulator AutoLongSimulator --n_mc 100 --n_steps 1000  --from_sigma -2 --to_sigma 0 --n_values 50 --model LongVariance --network InverseUNetVarianceEmbedding --max_hidden_size 2048 --activation relu --objective Novel --displacement True --checkpoint logs/SkullLandmarks_KunitaLong/version_${VERSION}/checkpoints/  | sed -e "s/^/Vulpes vulpes,/" >> $FILE

Rscript paper-plots/variance_skull.R $FILE
