from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset, S2T_Dataset_YTASL
import os
import argparse
from config import *

def main(args):

    print(args)
    utils.set_seed(args.seed)

    print(f"Creating dataset:")

    if args.dataset == "YTASL":
        train_data = S2T_Dataset_YTASL(path=train_label_paths[args.dataset],
                                 args=args, phase='train')
    else:
        train_data = S2T_Dataset(path=train_label_paths[args.dataset],
                                 args=args, phase='train')
    print(train_data)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=train_data.collate_fn,
                                  pin_memory=args.pin_mem,
                                  drop_last=True)

    model = Uni_Sign(
        args=args
    )
    # model.cuda()

    src_input, tgt_input = next(iter(train_dataloader))
    out = model(src_input, tgt_input)
    print(out)

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    main(args)