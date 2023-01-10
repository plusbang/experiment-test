from bigdl.nano.common.cpu_schedule import schedule_processors
import os
import json
import shutil
from tempfile import TemporaryDirectory
from contextlib import closing
import socket
import tensorflow as tf
from bigdl.nano.utils.log4Error import invalidInputError

from bigdl.nano.tf.keras.distributed_utils import _find_free_port
from bigdl.nano.common.multiprocessing.backend import Backend
from typing import Any


class MultiprocessingBackend(Backend):

    def setup(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def run(self, target, args=..., nprocs=1, envs=None) -> Any:
        if envs is not None:
            if isinstance(envs, list):
                invalidInputError(nprocs == len(envs),
                                  "envs must have the same length with nprocs")
            elif isinstance(envs, dict):
                envs = [envs] * nprocs
            else:
                invalidInputError(False, "envs must be a dict or a list of dict")

        return self.run_subprocess(target, args=args, nprocs=nprocs, envs=envs)

    def run_subprocess(self, target, args=..., nprocs=1, envs=None) -> Any:
        import cloudpickle
        import skops.io as sio
        import subprocess
        import sys

        with TemporaryDirectory() as temp_dir:
            # with open(os.path.join(temp_dir, "args.pkl"), 'wb') as f:
            #     cloudpickle.dump(args, f)
            # with open(os.path.join(temp_dir, "target.pkl"), 'wb') as f:
            #     cloudpickle.dump(target, f)
            sio.dump(args, os.path.join(temp_dir, "args.pkl"))
            sio.dump(target, os.path.join(temp_dir, "target.pkl"))

            ex_list = []
            cwd_path = os.path.dirname(__file__)
            print(cwd_path)
            for i in range(nprocs):
                for key, val in os.environ.items():
                    if key not in envs[i]:
                        envs[i][key] = val
                ex_list.append(subprocess.Popen([sys.executable, f"{cwd_path}/subprocess_worker.py",
                                                 temp_dir], env=envs[i]))
            for _, ex in enumerate(ex_list):
                ex.wait()

            results = []
            for i in range(nprocs):
                # with open(os.path.join(temp_dir, f"history_{i}"), "rb") as f:
                #     results.append(cloudpickle.load(f))
                results.append(sio.load(os.path.join(temp_dir, f"history_{i}"), trusted=True))
        return results


#(train_func, temp_dir, train_ds_def, train_elem_spec, fit_kwargs)
def _train_func(train_func, model_dir, ds_graph, elem_spec, fit_kwargs):
    import tensorflow as tf
    from tensorflow.python.distribute.coordinator.values import deserialize_dataset_from_graph

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        new_model = tf.keras.models.load_model(os.path.join(model_dir, "temp_model"))
        train_dataset = deserialize_dataset_from_graph(ds_graph, elem_spec)

        task_id = strategy.cluster_resolver.task_id

        if task_id == 0:
            verbose = fit_kwargs['verbose']
        else:
            verbose = 0
        del fit_kwargs['verbose']
        history = new_model.train_func(train_dataset)
        if task_id == 0:
            path = os.path.join(model_dir, 'trained_model_weights')
            new_model.save_weights(path, overwrite=True)
        else:
            path = os.path.join(model_dir, f'trained_model_weights_{task_id}')
            new_model.save_weights(path, overwrite=True)
        return history


def distributed_train_keras(backend, model, train_func, nprocs, fit_kwargs):
    """Run pseudo-distributed training on the keras model with the specified backend."""
    backend.setup()

    envs = schedule_processors(nprocs)

    from tensorflow.python.distribute.coordinator.values import serialize_dataset_to_graph

    train_dataset = fit_kwargs.pop('x')
    
    train_ds_def = serialize_dataset_to_graph(train_dataset).numpy()
    train_elem_spec = train_dataset.element_spec

    ports = set()
    while len(ports) < nprocs:
        ports.add(_find_free_port())
    ports = list(ports)
    worker_list = [f"localhost:{p}" for p in ports]

    with TemporaryDirectory() as temp_dir:
        model.save(os.path.join(temp_dir, 'temp_model'))

        for i, env in enumerate(envs):
            env.update({
                "TF_CONFIG": json.dumps(
                    {
                        'cluster': {
                            'worker': worker_list
                        },
                        'task': {'type': 'worker', 'index': i}
                    }),
                'no_proxy': "localhost"
            })

        train_args = (train_func, temp_dir, train_ds_def, train_elem_spec, fit_kwargs)

        histrories = backend.run(target=_train_func,
                                 args=train_args,
                                 nprocs=nprocs,
                                 envs=envs)
        model.load_weights(os.path.join(temp_dir, 'trained_model_weights'))
    return histrories[0]

