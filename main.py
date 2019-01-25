import argparse, os
from CNN import CNN

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


''' parsing and configuration '''
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot_dir', type=str, default='./data/', help='Root path of data')
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    parser.add_argument('--sample_num', type=int, default=64, help='The number of samples to test')
    parser.add_argument('--save_dir', type=str, default='./model/', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='./results/', help='Directory name to save the result')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=str_to_bool, default='True')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of threads')
    parser.add_argument('--comment', type=str, default='', help='Comment to pyt on model_name')

    parser.add_argument('--type', type=str, default='train', help='train or test')

    return check_args(parser.parse_args())


''' checking arguments '''
def check_args(opts):
    # --save_dir
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    # --result_dir
    if not os.path.exists(opts.result_dir):
        os.makedirs(opts.result_dir)

    # --comment
    if len(opts.comment) > 0:
        print("comment: " + opts.comment)
        comment_part = '_' + opts.comment
    else:
        comment_part = ''
    tempconcat = "model" + comment_part
    print('model and loss plot -> ' + os.path.join(opts.save_dir, tempconcat))

    # --epoch
    try:
        assert opts.epoch >= 1
    except:
        print("number of epoch must be larger than or equal to one")

    # --batch_size
    try:
        assert opts.batch_size >= 1
    except:
        print("batch size must be larger than or equal to one")

    print(opts)

    return opts


''' main '''
def main():
    # parse arguments
    opts = parse_args()
    if opts is None:
        print("There is no opts!!")
        exit()

    model0 = CNN(opts, '', 6)
    model1 = CNN(opts, '0/', 3)
    model2 = CNN(opts, '1/', 3)

    if opts.type == 'train':
        print("[*] Training started")
        model0.train(0)
        model1.train(1)
        model2.train(2)
        print("[*] Training finished")

    elif opts.type == 'test':
        print("[*] Test started")
        model0.test([model1, model2])
        print("[*] Test finished")


if __name__ == '__main__':
    main()
