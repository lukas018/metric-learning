#!/usr/bin/env python3
#
from pathlib import Path

import yaml

from .algorithms.lightning.metric_module import PreTrainArguments, FewshotArguments


def parse_arg_file(path_to_args: str):
        args_dict = yaml.load(stream=Path(path_to_args).read_text(), Loader=yaml.BaseLoader)

        pretrain_args = PreTrainArguments(**args_dict.get('pre_training', {}))
        fs_args = FewshotArguments(**args_dict.get('fewshot_training', {}))

        return pretrain_args, fs_args
