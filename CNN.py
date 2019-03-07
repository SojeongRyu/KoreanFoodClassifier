''' CNN AlexNet '''

import torch, time, os, pickle
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
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


class CNN(object):
    def __init__(self, args):
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
        self.resl = 256
        self.num_cls = 10
        self.crop_size = 227
        self.total_correct = 0
        self.total_test_len = 0

        # construct model
        self.net = Net(self.num_cls)
        # define optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        if self.gpu_mode:
            self.net = self.net.cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.CE_loss = nn.CrossEntropyLoss()

        # load dataset
        if self.type == 'crossvalidation':
            data_transform = transforms.Compose([
                transforms.Resize(self.resl),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor()
            ])

            dataset = {
            x: datasets.ImageFolder(root=os.path.join(self.dataroot_dir, (x + '/')), transform=data_transform)
            for x in ['train', 'test']}

            kf = KFold(n_splits=5, shuffle=True)

            for i, (train_index, test_index) in enumerate(kf.split(dataset['train'])):
                train_set = Subset(dataset['train'], train_index)
                test_set = Subset(dataset['train'], test_index)
                trainloader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                         num_workers=self.num_workers, pin_memory=False)
                testloader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True,
                                         num_workers=self.num_workers, pin_memory=False)

                print('\n\n**********[Fold : {}, train : {}, test : {}]**********'.format(i + 1, len(trainloader.dataset), len(testloader.dataset)))

                self.train(i + 1, trainloader)
                self.test(i + 1, testloader)

            print("Average Accuracy: %d/%d (%f)" % (self.total_correct, self.total_test_len, 100. * (self.total_correct / self.total_test_len)))

        else:
            # load dataset
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(self.resl),
                    transforms.RandomCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ]),
                'test': transforms.Compose([
                    transforms.Resize(self.resl),
                    transforms.CenterCrop(self.crop_size),
                    transforms.ToTensor()
                ]),
            }
            dataset = {
                x: datasets.ImageFolder(root=os.path.join(self.dataroot_dir, (x + '/')), transform=data_transforms[x])
                for x in ['train', 'test']}
            self.dataloaders = {
                x: DataLoader(dataset[x], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
                for x in ['train', 'test']}


    def train(self, fold_num = -1, trainloader = None):
        self.train_hist = {}
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        # train
        self.net.train()
        start_time = time.time()
        if trainloader == None:
            trainloader = self.dataloaders['train']
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
            # self.visualize_result(epoch, self.z)
            # ----check train.result---- #
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            utils.loss_plot(self.train_hist, self.result_dir, comment=self.comment, fold_num = fold_num)
            self.save(fold_num)
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Training finished!... save training results")
        print("Total time:", self.train_hist['total_time'][0])
        print("Per epoch time:", self.train_hist['per_epoch_time'], '\n')
        self.save(fold_num)

    def save(self, fold_num):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if fold_num == -1:
            torch.save(self.net.state_dict(),
                       os.path.join(self.save_dir, 'model_' + self.comment  + '.pkl'))
            with open(os.path.join(self.save_dir, 'history_' + self.comment  + '.pkl'), 'wb') as f:
                pickle.dump(self.train_hist, f)
        else:
            torch.save(self.net.state_dict(), os.path.join(self.save_dir, 'model_' + self.comment + '_fold' + str(fold_num) + '.pkl'))
            with open(os.path.join(self.save_dir, 'history_' + self.comment + '_fold' + str(fold_num) + '.pkl'), 'wb') as f:
                pickle.dump(self.train_hist, f)

    def load(self, fold_num):
        if fold_num == -1:
            self.net.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_' + self.comment + '.pkl')))
        else:
            self.net.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_' + self.comment + '_fold' + str(fold_num) + '.pkl')))

    def test(self, fold_num=-1, testloader=None):
        self.net.eval()
        self.load(fold_num)

        if testloader == None:
            testloader = self.dataloaders['test']

        test_loss, correct = 0, 0
        error_cnt = [[0 for x in range(self.num_cls)] for y in range(self.num_cls)]
        with torch.no_grad():
            for img, target in testloader:
                if self.gpu_mode:
                    img, target = Variable(img.cuda()), Variable(target.cuda())
                else:
                    img, target = Variable(img), Variable(target)

                output = self.net(img)
                test_loss += self.CE_loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                for i in range(len(target)):
                    error_cnt[target[i].item()][pred.view_as(target)[i].item()] += 1
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(testloader.dataset)

        self.total_correct += correct
        self.total_test_len += len(testloader.dataset)
        print('Average loss: %f' % test_loss,
              "Accuracy: %d/%d (%f)" % (correct, len(testloader.dataset),
                                        100. * correct / len(testloader.dataset)))
        print('error tracking')
        for i in range(self.num_cls):
            print('class %d:' % i, error_cnt[i],
                  '\taccuracy: %f(%d/%d)' % (error_cnt[i][i] / sum(error_cnt[i]) * 100, error_cnt[i][i], sum(error_cnt[i])))

    def predict(self, img, fold_num=-1):
        self.net.eval()
        self.load(fold_num)

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
