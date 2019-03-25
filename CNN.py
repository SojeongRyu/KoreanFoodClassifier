''' CNN AlexNet '''

import torch, time, os, pickle
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import utils
import operator
import warnings
warnings.filterwarnings('ignore')


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.features = nn.Sequential(  # input size=227*227
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        utils.initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 6 * 6 * 256)
        x = self.classifier(x)
        return x


class SubNet(nn.Module):
    def __init__(self, num_classes):
        super(SubNet, self).__init__()
        self.features = nn.Sequential(  # input size=227*227
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6 * 6 * 128, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

        utils.initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 6 * 6 * 128)
        x = self.classifier(x)
        return x


class CNN(object):
    def __init__(self, args, sub_net_name=''):
        # parameters
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataroot_dir = args.dataroot_dir
        self.sample_num = args.sample_num
        self.gpu_mode = args.gpu_mode
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.comment = args.comment
        self.type = args.type
        self.fold_num = args.fold_num
        self.resl = 256
        self.num_cls = 10
        self.crop_size = 227
        self.total_correct = 0
        self.total_test_len = 0
        self.index = 0

        # construct model
        if len(sub_net_name) == 0:
            self.net = Net(self.num_cls)
        else:
            self.net = Net(2)
        # define optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        if self.gpu_mode:
            self.net = self.net.cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.CE_loss = nn.CrossEntropyLoss()

    def train(self, fold_num=-1, trainloader=None, sub_net_name=''):
        self.train_hist = {}
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        # train
        self.net.train()

        start_time = time.time()
        if trainloader is None:
            # load dataset
            data_transform = transforms.Compose([
                transforms.Resize(self.resl),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
            if len(sub_net_name) == 0:
                dataset = datasets.ImageFolder(root=os.path.join(self.dataroot_dir, 'train/'), transform=data_transform)
            else:
                dataset = datasets.ImageFolder(root=os.path.join(self.dataroot_dir, sub_net_name + '/'), transform=data_transform)
            trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            for iB, (img_, label_) in enumerate(trainloader):
                if iB == trainloader.dataset.__len__() // self.batch_size:
                    break
                if self.gpu_mode:
                    label, img_ = Variable(label_.cuda()), Variable(img_.cuda())
                else:
                    label, img_ = Variable(label_), Variable(img_)
                # ----Update cnn_network---- #
                self.optimizer.zero_grad()
                output = self.net(img_)
                loss = self.CE_loss(output, label)
                self.train_hist['G_loss'].append(loss.item())
                loss.backward()
                self.optimizer.step()
                # ----check train result---- #
                if (iB % 100 == 0) and (epoch % 1 == 0):
                    print('[E%03d]' % (epoch) + '\tloss: %.6f' % (loss.item()))
            # ----check train.result---- #
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            utils.loss_plot(self.train_hist, self.result_dir, comment=self.comment, fold_num=fold_num, sub_net_name=sub_net_name)
            self.save(fold_num, sub_net_name)
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Training finished!... save training results")
        print("Total time:", self.train_hist['total_time'][0])
        print("Per epoch time:", self.train_hist['per_epoch_time'], '\n')
        self.save(fold_num, sub_net_name)

    def save(self, fold_num=-1, sub_net_name=''):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        fold_info = ''
        if fold_num != -1:
            fold_info = '_fold' + str(fold_num)
        if len(sub_net_name) > 0:
            sub_net_name = '_' + sub_net_name

        torch.save(self.net.state_dict(), os.path.join(self.save_dir, 'model_' + self.comment + fold_info + sub_net_name + '.pkl'))
        with open(os.path.join(self.save_dir, 'history_' + self.comment + fold_info + sub_net_name + '.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self, fold_num=-1, sub_net_name=''):
        fold_info = ''
        if fold_num != -1:
            fold_info = '_fold' + str(fold_num)
        if len(sub_net_name) > 0:
            sub_net_name = '_' + sub_net_name

        self.net.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_' + self.comment + fold_info + sub_net_name + '.pkl')))

    def test(self, fold_num=-1, testloader=None, sub_model=None, sub_net_name=''):
        self.net.eval()
        self.load(fold_num)
        if sub_model is not None:
            sub_model.net.eval()
            sub_model.load(sub_net_name=sub_net_name)

        if testloader is None:
            # load dataset
            data_transform = transforms.Compose([
                transforms.Resize(self.resl),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor()
            ])
            dataset = datasets.ImageFolder(root=os.path.join(self.dataroot_dir, 'test/'), transform=data_transform)
            testloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        test_loss, correct, correct_top2, correct_subnet = 0, 0, 0, 0
        error_cnt = [[0 for x in range(self.num_cls)] for y in range(self.num_cls)]
        error_cnt_top2 = [[0 for x in range(self.num_cls)] for y in range(self.num_cls)]
        error_cnt_subnet = [[0 for x in range(self.num_cls)] for y in range(self.num_cls)]
        subnet_classes = list(map(int, sub_net_name.split('+')))

        with torch.no_grad():
            for img, target in testloader:
                if self.gpu_mode:
                    img, target = Variable(img.cuda()), Variable(target.cuda())
                else:
                    img, target = Variable(img), Variable(target)

                output = self.net(img)
                test_loss += self.CE_loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                pred_subnet = pred.clone()
                pred_top2 = output.clone()
                for i in range(len(pred_top2)):
                    pred_top2[i][pred[i].item()] = -float('inf')
                pred_top2 = pred_top2.argmax(dim=1, keepdim=True)

                for i in range(len(target)):
                    error_cnt[target[i].item()][pred.view_as(target)[i].item()] += 1
                    if target[i] == pred_top2[i]:
                        error_cnt_top2[target[i].item()][target[i].item()] += 1
                    else:
                        error_cnt_top2[target[i].item()][pred.view_as(target)[i].item()] += 1
                    if pred_subnet.view_as(target)[i].item() in subnet_classes:
                        tmp_output = sub_model.net(img[i].expand(1, 3, 227, 227))
                        pred_subnet[i][0] = subnet_classes[tmp_output.argmax(dim=1, keepdim=True)[0][0].item()]
                    error_cnt_subnet[target[i].item()][pred_subnet.view_as(target)[i].item()] += 1
                correct += pred.eq(target.view_as(pred)).sum().item()
                correct_top2 += pred.eq(target.view_as(pred)).sum().item() + pred_top2.eq(target.view_as(pred)).sum().item()
                correct_subnet += pred_subnet.eq(target.view_as(pred)).sum().item()
        test_loss /= len(testloader.dataset)

        self.total_correct += correct
        self.total_test_len += len(testloader.dataset)

        print('Average loss: %f' % test_loss)
        self.print_test_result(correct, len(testloader.dataset), error_cnt, '')
        self.print_test_result(correct_top2, len(testloader.dataset), error_cnt_top2, 'top 2')
        self.print_test_result(correct_subnet, len(testloader.dataset), error_cnt_subnet, 'subnet ' + sub_net_name)

    def print_test_result(self, correct, datacnt, error_cnt, comment):
        print('Accuracy '+ comment + ': %d/%d (%f)' % (correct, datacnt, 100. * correct / datacnt))
        print('Error tracking', comment)
        for i in range(self.num_cls):
            print('class %d:' % i, error_cnt[i],
                  '\taccuracy: %f(%d/%d)' % (error_cnt[i][i] / sum(error_cnt[i]) * 100, error_cnt[i][i], sum(error_cnt[i])))

    def predict(self, img, fold_num=-1, sub_net_name=''):
        self.net.eval()
        self.load(fold_num, sub_net_name)

        data_transform = transforms.Compose([
                transforms.Resize(self.resl),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor()
        ])

        input = data_transform(img)
        output = self.net(input.expand(1, 3, 227, 227))

        file = open(self.dataroot_dir + '/class.txt')
        classes = file.readlines()[1:]
        for i in range(len(classes)):
            classes[i] = classes[i].split()
            classes[i][1].strip()
        print(classes)
        sm = torch.nn.Softmax()
        output_list = sm(output).tolist()[0]
        output_dict = dict()
        for i in range(len(output_list)):
            output_dict[int(classes[i][0])] = output_list[i]
            form = '{:.5%}'.format(output_list[i])
            print('[%2d] ' % int(classes[i][0]), '%-12s' % classes[i][1], '\t', form, sep='')

        output_sorted = sorted(output_dict.items(), key=operator.itemgetter(1), reverse=True)
        return output_sorted[0:2]

    def plot_kernels(self, tensor, num_cols=10):
        num_rows = 10
        fig = plt.figure(figsize=(num_cols, num_rows))
        self.index = 0

        # Normalise
        maxVal = tensor.max()
        minVal = abs(tensor.min())
        maxVal = max(maxVal, minVal)
        tensor = tensor / maxVal
        tensor = tensor / 2
        tensor = tensor + 0.5
        for t in tensor:
            ax1 = fig.add_subplot(num_rows, num_cols, self.index + 1)
            pilTrans = transforms.ToPILImage()
            pilImg = pilTrans(t)
            ax1.imshow(pilImg, interpolation='none')
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            self.index += 1

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()

    def display_weights(self, fold_num=-1, sub_net_name=''):
        self.load(fold_num=fold_num, sub_net_name=sub_net_name)
        self.index = 0
        fold_info = ''
        if fold_num != -1:
            fold_info = '_fold' + str(fold_num)
        if len(sub_net_name) > 0:
            sub_net_name = '_' + sub_net_name
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                if self.index == 0:
                    self.plot_kernels(m.weight.data.clone().cpu())
                    plt.savefig(os.path.join(self.result_dir, 'weight_' + self.comment + fold_info + sub_net_name + '.png'))


def cross_validation(args):
    # load dataset
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor()
    ])

    total_correct = 0
    total_test_len = 0

    dataset = {
        x: datasets.ImageFolder(root=os.path.join(args.dataroot_dir, (x + '/')), transform=data_transform)
        for x in ['train', 'test']}

    kf = KFold(n_splits=args.fold_num, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(dataset['train'])):
        model = CNN(args)

        train_set = Subset(dataset['train'], train_index)
        test_set = Subset(dataset['train'], test_index)
        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
        testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

        print('\n\n**********[Fold : {}, train : {}, test : {}]**********'
              .format(i + 1, len(trainloader.dataset), len(testloader.dataset)))
        model.train(i + 1, trainloader)
        model.test(i + 1, testloader)
        total_correct += model.total_correct
        total_test_len += model.total_test_len

    print("Average Accuracy: %d/%d (%f)"
          % (total_correct, total_test_len, 100. * (total_correct / total_test_len)))
