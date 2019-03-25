import argparse, os
from PIL import Image
from CNN import CNN, cross_validation
from torchvision.utils import make_grid

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
    parser.add_argument('--epoch', type=int, default=70, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    parser.add_argument('--sample_num', type=int, default=64, help='The number of samples to test')
    parser.add_argument('--save_dir', type=str, default='./model/', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='./results/', help='Directory name to save the result')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=str_to_bool, default='True')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of threads')
    parser.add_argument('--comment', type=str, default='', help='Comment to pyt on model_name')
    parser.add_argument('--type', type=str, default='train', help='train or test or pred or cv or vis')
    parser.add_argument('--fold_num', type=int, default=-1, help='fold number')
    parser.add_argument('--sub_net_name', type=str, default='', help='the name of subnet')

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

    main_model = CNN(opts)
    sub_model = CNN(opts, opts.sub_net_name)
    if opts.type == 'train':
        print("[*] Training started")
        main_model.train(fold_num=opts.fold_num)
        sub_model.train(sub_net_name=opts.sub_net_name)
        print("[*] Training finished")

    elif opts.type == 'test':
        print("[*] Test started")
        main_model.test(fold_num=opts.fold_num, sub_model=sub_model, sub_net_name=opts.sub_net_name)
        print("[*] Test finished")

    elif opts.type == 'cv':
        print("[*] cv started")
        cross_validation(opts)
        print("[*] cv finished")

    elif opts.type == 'pred':
        print('Predict result:', img=main_model.predict(Image.open('./img.jpg')), fold_num=opts.fold_num)

    elif opts.type == 'vis':
        main_model.display_weights(fold_num=opts.fold_num)
        sub_model.display_weights(sub_net_name=opts.sub_net_name)


if __name__ == '__main__':
    main()
