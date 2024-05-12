from unittest import mock

import pytest

from thesis.scripts import common


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_points_1d(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Model'.split() +
        '--constraints PointConstraints --initial 0 --terminal 1'.split() +
        '--diffusion Brownian1D --variance 1'.split() +
        f'--simulator LongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_points_2d(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Model'.split() +
        '--constraints PointConstraints --initial 1,0 --terminal 2,1.5'.split() +
        '--diffusion Brownian2D --variance 1 --covariance 0.5'.split() +
        f'--simulator LongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_mixture_2d(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Model'.split() +
        '--constraints PointMixtureConstraints --initial_a 1,1 --initial_b 3,3 --terminal 0,0'.split() +
        '--diffusion Brownian2D --variance 0.5 --covariance 0'.split() +
        f'--simulator LongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_points_nd(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Model'.split() +
        '--constraints PointConstraints --initial 1.0,2.0 --terminal 2.0,4.0'.split() +
        '--diffusion BrownianND --d 2 --variance 1'.split() +
        f'--simulator LongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_points_2d_wide(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Model'.split() +
        '--constraints PointConstraints2D --initial 0,2,2,0 --terminal 0,0.2,0.2,0'.split() +
        '--diffusion KunitaLong --variance 1 --gamma 0.5'.split() +
        f'--simulator LongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_circle_brownian_nd(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Model'.split() +
        '--constraints CircleLandmarks --k 16 --initial_radius 1 --terminal_radius 2.5 --skewness 1.5'.split() +
        '--diffusion BrownianND --d 32 --variance 0.1'.split() +
        f'--simulator LongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_circle_brownian_wide(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Factorised'.split() +
        '--constraints CircleLandmarks --k 16 --initial_radius 1 --terminal_radius 2.5 --skewness 1.5'.split() +
        '--diffusion BrownianWideKernel --variance 0.2 --gamma 0.5'.split() +
        f'--simulator AutoLongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_circle_brownian_long(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Model'.split() +
        '--constraints CircleLandmarks --k 16 --initial_radius 1 --terminal_radius 2.5 --skewness 1.5'.split() +
        '--diffusion BrownianLongKernel --variance 0.2 --gamma 0.5'.split() +
        f'--simulator LongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_circle_kunita_wide(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Factorised'.split() +
        '--constraints CircleLandmarks --k 16 --initial_radius 1 --terminal_radius 2.5 --skewness 1.5'.split() +
        '--diffusion KunitaWide --variance 0.2 --gamma 0.5'.split() +
        f'--simulator AutoLongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_circle_kunita_long(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Model'.split() +
        '--constraints CircleLandmarks --k 20 --initial_radius 2 --terminal_radius 2.5 --skewness 1.5'.split() +
        '--diffusion KunitaLong --variance 0.2 --gamma 0.5'.split() +
        f'--simulator LongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_butterfly_brownian_wide(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Factorised'.split() +
        '--constraints ButterflyLandmarks --initial_butterfly ../data/butterflies/landmarks/honrathi_pts.npy --terminal_butterfly ../data/butterflies/landmarks/polytes_pts.npy --every 3'.split() +
        '--diffusion BrownianWideKernel --variance 0.1 --gamma 0.02'.split() +
        f'--simulator AutoLongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_ball_brownian_wide(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Factorised'.split() +
        '--constraints BallLandmarks --k 40 --initial_radius 1 --terminal_radius 2 --skewness 1.5'.split() +
        '--diffusion BrownianWideKernel --variance 0.1 --gamma 0.02'.split() +
        f'--simulator AutoLongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_skull_brownian_wide(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Factorised'.split() +
        '--constraints SkullLandmarks --landmarks_info ../data/canidae/skull_landmarks_information.csv --initial_skull ../data/canidae/landmarks/al_Canislupus_Bergen_B2.csv --terminal_skull ../data/canidae/landmarks/al_Canislupus_Bergen2698.csv --every 3 --bone 9'.split() +
        '--diffusion BrownianWideKernel --variance 0.1 --gamma 0.02'.split() +
        f'--simulator AutoLongSimulator --displacement {displacement}'.split()
    )


@pytest.mark.parametrize('displacement', ('True', 'False'))
def test_points_1d_conditioned(tmp_path, displacement):
    _run(
        tmp_path,
        '--model Model'.split() +
        '--constraints PointConstraints --initial 0 --terminal 1'.split() +
        '--diffusion Brownian1D --variance 1'.split() +
        f'--simulator LongSimulator --displacement {displacement}'.split() +
        '--min_diffusion_scale 0.5 --max_diffusion_scale 2'.split()
    )


def _run(tmp_path, args):
    with mock.patch(
        'thesis.lightning.loggers.Logger.__init__.__defaults__',
        (str(tmp_path), 'default', None),
    ):
        print(args)
        with mock.patch(
            'sys.argv',
            [
                '--rng_key',
                '42',
            ] +
            args +
            [
                '--n',
                '16',
                '--epochs',
                '1'
            ],
        ):
            common.main()
