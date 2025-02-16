POINTS=$(python paper-plots/brownian_2d_points_collection.py --rng_key 42 --diffusion Brownian2D --variance 1 --covariance 0 --simulator LongSimulator --n_points 10)

python thesis/inference/diffusion_mean.py --rng_key 42 --constraints PointConstraints2DCollection --initial_points " $POINTS" --diffusion BrownianNDWide --d 2 --variance 1 --sigma 1 --simulator AutoLongSimulator --n_steps 15 --update_rate 0.05

python paper-plots/visualise_brownian_diffusion_mean.py
