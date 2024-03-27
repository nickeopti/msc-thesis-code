import abc
import csv
import pathlib


class Logger(abc.ABC):
    def __init__(self, save_dir: str = 'logs', name: str = 'default', version: int = None) -> None:
        self.save_dir = pathlib.Path(save_dir)
        self.name = name
        self._version = version

        self._has_written_header = False

    @abc.abstractmethod
    def log(self, epoch: int, train_loss, validation_loss):
        ...

    @property
    def path(self) -> pathlib.Path:
        return self.save_dir / self.name / f'version_{self.version}'

    @property
    def version(self):
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self) -> int:
        root_dir = self.save_dir / self.name

        if not root_dir.is_dir():
            return 0

        existing_versions = []
        for d in root_dir.iterdir():
            if d.is_dir() and d.name.startswith('version_'):
                existing_versions.append(int(d.name.split('_')[-1]))

        return max(existing_versions, default=-1) + 1


class CSVLogger(Logger):
    def log(self, epoch: int, train_loss, validation_loss):
        with open(self.path / 'metrics.csv', 'a' if self._has_written_header else 'w') as f:
            writer = csv.DictWriter(f, ('epoch', 'train', 'validation'))

            if not self._has_written_header:
                writer.writeheader()
                self._has_written_header = True

            writer.writerow(
                {
                    'epoch': epoch,
                    'train': train_loss,
                    'validation': validation_loss,
                }
            )
