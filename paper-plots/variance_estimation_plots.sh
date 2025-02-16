FILE=variance_methods.csv

echo "steps,landmarks,method,sigma,ll" > $FILE

for STEPS in 10 100 1000; do
    python thesis/inference/variance.py --rng_key 42 --constraints ButterflyLandmarks --data_path ../data/butterflies/aligned_dataset_n43/ --initial_species "Papilio ambrax" --terminal_species "Papilio slateri" --every 30 --diffusion BrownianWideKernel --variance 1 --gamma 0.005 --simulator AutoLongSimulator --n_mc 100 --n_steps $STEPS  --from_sigma -2 --to_sigma -0.5 --n_values 50 | sed -e "s/^/${STEPS},20,/" >> $FILE

    python thesis/inference/variance.py --rng_key 42 --constraints ButterflyLandmarks --data_path ../data/butterflies/aligned_dataset_n43/ --initial_species "Papilio ambrax" --terminal_species "Papilio slateri" --every 12 --diffusion BrownianWideKernel --variance 1 --gamma 0.005 --simulator AutoLongSimulator --n_mc 100 --n_steps $STEPS  --from_sigma -2 --to_sigma -0.5 --n_values 50 | sed -e "s/^/${STEPS},50,/" >> $FILE

    # python thesis/inference/variance.py --rng_key 42 --constraints ButterflyLandmarks --data_path ../data/butterflies/aligned_dataset_n43/ --initial_species "Papilio ambrax" --terminal_species "Papilio slateri" --every 3 --diffusion BrownianWideKernel --variance 1 --gamma 0.005 --simulator AutoLongSimulator --n_mc 100 --n_steps $STEPS  --from_sigma -2 --to_sigma -0.5 --n_values 50 | sed -e "s/^/${STEPS},200,/" >> $FILE

    python thesis/inference/variance.py --rng_key 42 --constraints ButterflyLandmarks --data_path ../data/butterflies/aligned_dataset_n43/ --initial_species "Papilio ambrax" --terminal_species "Papilio slateri" --every 1 --diffusion BrownianWideKernel --variance 1 --gamma 0.005 --simulator AutoLongSimulator --n_mc 100 --n_steps $STEPS  --from_sigma -2 --to_sigma -0.5 --n_values 50 | sed -e "s/^/${STEPS},600,/" >> $FILE
done

Rscript paper-plots/variance_methods.R $FILE 
