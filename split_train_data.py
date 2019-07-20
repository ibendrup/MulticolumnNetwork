import argparse 
import os
import sys

import pandas as pd
import numpy as np


def main(args):

    train_df = pd.read_csv(os.path.join(args.input_dir, 'train_df.csv'))
    train_descr_arr = np.load(os.path.join(args.input_dir, 'train_df_descriptors.npy'))
    train_track_order_df = pd.read_csv(os.path.join(args.input_dir, 'train_df_track_order_df.csv'))
    train_gt_df = pd.read_csv(os.path.join(args.input_dir, 'train_gt_df.csv'))

    dev_person_ids = train_df[train_df.is_val == True].person_id.unique()
    train_person_ids = train_df[train_df.is_val == False].person_id.unique()

    dev_descr_arr = train_descr_arr[train_df.is_val == True]
    train_descr_arr = train_descr_arr[train_df.is_val == False]
    dev_df = train_df[train_df.is_val == True]
    train_df = train_df[train_df.is_val == False]

    dev_track_order_df = train_track_order_df[train_track_order_df.person_id.isin(dev_person_ids)]
    train_track_order_df = train_track_order_df[train_track_order_df.person_id.isin(train_person_ids)]

    train_df.to_csv(os.path.join(args.output_dir, 'train.csv'))
    dev_df.to_csv(os.path.join(args.output_dir, 'dev.csv'))
    train_track_order_df.to_csv(os.path.join(args.output_dir, 'train_track_order.csv'))
    dev_track_order_df.to_csv(os.path.join(args.output_dir, 'dev_track_order.csv'))
    np.save(os.path.join(args.output_dir, 'train_emb.npy'), train_descr_arr)
    np.save(os.path.join(args.output_dir, 'dev_emb.npy'), dev_descr_arr)

    gt_df = pd.read_csv(os.path.join(args.input_dir, 'train_gt_df.csv'))
    gt_descr_arr = np.load(os.path.join(args.input_dir, 'train_gt_descriptors.npy'))

    dev_gt_df = gt_df[gt_df.person_id.isin(dev_person_ids)]
    train_gt_df = gt_df[gt_df.person_id.isin(train_person_ids)]
    dev_gt_descr_arr = gt_descr_arr[gt_df.person_id.isin(dev_person_ids)]
    train_gt_descr_arr = gt_descr_arr[gt_df.person_id.isin(train_person_ids)]

    train_gt_df.to_csv(os.path.join(args.output_dir, 'train_gt.csv'))
    dev_gt_df.to_csv(os.path.join(args.output_dir, 'dev_gt.csv'))
    np.save(os.path.join(args.output_dir, 'train_gt_emb.npy'), train_gt_descr_arr) 
    np.save(os.path.join(args.output_dir, 'dev_gt_emb.npy'), dev_gt_descr_arr)

    return

   

def parse_arguments(argv):
    """Parse command line arguments
    """
    parser = argparse.ArgumentParser()
        
    #model params
    parser.add_argument('--input_dir', type=str,
        help='Input directory.')
    parser.add_argument('--output_dir', type=str,
        help='Output directory.')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
