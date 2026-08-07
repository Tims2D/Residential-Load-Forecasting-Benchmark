from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {'household_data_1min': Dataset_Custom,
             '10seconds_load': Dataset_Custom,
             '15Minute_load': Dataset_Custom,
             'hourly_load': Dataset_Custom,
            }

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    output_attn_map = args.output_attn_map

    if flag == 'test':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        output_attn_map = args.output_attn_map
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            output_attn_map=args.output_attn_map,
            data_name=args.data_name
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

                
