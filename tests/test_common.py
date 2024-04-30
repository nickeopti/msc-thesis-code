from unittest import mock

from thesis.scripts import common


def test_common(tmp_path):
    with mock.patch(
        'thesis.lightning.loggers.Logger.__init__.__defaults__',
        (str(tmp_path), 'default', None),
    ):
        with mock.patch(
            'sys.argv', [
                '--rng_key',
                '42',
                '--model',
                'Factorised',
                '--constraints',
                'SkullLandmarks',
                '--landmarks_info',
                '../data/skull_landmarks_information.csv',
                '--initial_skull',
                '../data/al_Canislupus_Bergen_B2.csv',
                '--terminal_skull',
                '../data/al_Canislupus_Bergen2698.csv',
                '--every',
                '3',
                '--bone',
                '9',
                '--diffusion',
                'BrownianWideKernel',
                '--variance',
                '0.01',
                '--gamma',
                '0.001',
                '--simulator',
                'AutoLongSimulator',
                '--displacement',
                'True',
                '--n',
                '16',
                '--epochs',
                '1'
            ],
        ):
            common.main()
