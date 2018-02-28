import pathlib
from collections import defaultdict
import json
import numbers
from datetime import datetime
import numpy as np
from .miscs import get_git_hash


class Reporter(object):
    def __init__(self, save_dir):
        """
        base class of Reporter
        >>> with Reporter("save_dir") as r:
        >>>     r.add_scalar(1, "loss", idx=0)
        # automatically save the results
        """
        self._container = defaultdict(list)
        self._save_dir = save_dir
        self._now = datetime.now().strftime("%b%d-%H-%M-%S")
        self._filename = self._now + get_git_hash() + ".json"

    def add_scalar(self, x, name: str, idx: int):
        raise NotImplementedError

    def add_scalars(self, x: dict, name, idx: int):
        raise NotImplementedError

    def add_parameters(self, x, name: str, idx: int):
        raise NotImplementedError

    def add_text(self, x, name: str, idx: int):
        raise NotImplementedError

    def add_image(self, x, name: str, idx: int):
        raise NotImplementedError

    def add_images(self, x, name: str, idx: int):
        raise NotImplementedError

    def _register_data(self, x, name: str, idx: int):
        x = x if isinstance(x, str) else float(x)
        self._container[name].append((idx, x))

    def save(self):
        if self._save_dir:
            p = pathlib.Path(self._save_dir)
            p.mkdir()
            with (p / self._filename).open("w") as f:
                json.dump(self._container, f)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._save_dir is not None:
            self.save()

    @staticmethod
    def _tensor_type_check(x):
        if "Variable" in str(type(x)):
            x = x.data

        if "numpy" in str(type(x)):
            dim = x.ndim
        elif "Tensor" in str(type(x)):
            # should be torch.**Tensor
            x = x.cpu()
            dim = x.dim()

        return x, dim


class TQDMReporter(Reporter):
    def __init__(self, iterable, save_dir=None):
        """
        >>> with TQDMReporter(range(100)) as tqrange:
        >>>     for i in tqrange:
        >>>         pass
        """
        from tqdm import tqdm

        super(TQDMReporter, self).__init__(save_dir)
        self._tqdm = tqdm(iterable, ncols=100)

    def __iter__(self):
        for x in self._tqdm:
            yield x

    def add_scalar(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self._tqdm.set_postfix({name: x})

    def add_scalars(self, x: dict, name, idx: int):
        assert isinstance(x, dict)
        for k, v in x.items():
            self._register_data(v, k, idx)
        self._tqdm.set_postfix(x)


class VisdomReporter(Reporter):
    def __init__(self, port=6006, save_dir=None):
        from visdom import Visdom

        super(VisdomReporter, self).__init__(save_dir)
        self._viz = Visdom(port=port, env=self._now)
        self._lines = defaultdict()
        assert self._viz.check_connection(), f"""
        Please launch visdom.server before calling VisdomReporter.
        $python -m visdom.server -port {port}
        """

    def add_scalar(self, x, name: str, idx: int, **kwargs):
        self.add_scalars({name: x}, name=name, idx=idx, **kwargs)

    def add_scalars(self, x: dict, name, idx: int, **kwargs):
        x = {k: self._to_numpy(v) for k, v in x.items()}
        num_lines = len(x)
        is_new = self._lines.get(name) is None
        self._lines[name] = 1
        for k, v in x.items():
            self._register_data(v, k, idx)
        opts = dict(title=name,
                    legend=list(x.keys()))
        opts.update(**kwargs)
        X = np.column_stack((self._to_numpy(idx) for _ in range(num_lines)))
        Y = np.column_stack(x.values())
        self._viz.line(X=X, Y=Y, update=None if is_new else "append", win=name, opts=opts)

    def add_parameters(self, x, name: str, idx: int, **kwargs):
        # todo
        raise NotImplementedError

    def add_text(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self._viz.text(x)

    def add_image(self, x, name: str, idx: int):
        x, dim = self._tensor_type_check(x)
        assert dim == 3
        self._viz.image(self._normalize(x), opts=dict(title="name", caption=str(idx)))

    def add_images(self, x, name: str, idx: int):
        x, dim = self._tensor_type_check(x)
        assert dim == 4
        self._viz.images(self._normalize(x), opts=dict(title="name", caption=str(idx)))

    def _to_numpy(self, x):
        if isinstance(x, numbers.Number):
            x = np.array([x])
        elif "Tensor" in str(type(x)):
            x = x.numpy()
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(VisdomReporter, self).__exit__(exc_type, exc_val, exc_tb)
        self._viz.close()

    @staticmethod
    def _normalize(x):
        # normalize tensor values in (0, 1)
        _min, _max = x.min(), x.max()
        return (x - _min) / (_max - _min)


class TensorBoardReporter(Reporter):
    def __init__(self, save_dir=None):
        from tensorboardX import SummaryWriter

        super(TensorBoardReporter, self).__init__(save_dir)
        self._writer = SummaryWriter(log_dir=save_dir)

    def add_scalar(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self._writer.add_scalar(name, x, idx)

    def add_scalars(self, x: dict, name, idx: int):
        for k, v in x.items():
            self._register_data(v, k, idx)
        self._writer.add_scalars(name, x, idx)

    def add_image(self, x, name: str, idx: int):
        x, dim = self._tensor_type_check(x)
        assert dim == 3
        self._writer.add_image(name, x, idx)

    def add_text(self, x, name: str, idx: int):
        self._register_data(x, name, idx)
        self._writer.add_text(name, x, idx)

    def add_parameters(self, x, name: str, idx: int):
        self._writer.add_histogram(name, x, idx, bins="sqrt")

    def __exit__(self, exc_type, exc_val, exc_tb):
        super(TensorBoardReporter, self).__exit__(exc_type, exc_val, exc_tb)
        self._writer.close()
