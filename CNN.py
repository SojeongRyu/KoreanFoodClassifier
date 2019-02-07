''' CNN AlexNet '''

import torch, time, os, pickle
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import utils
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
        self.resl = 256
        self.num_cls = 10
        self.crop_size = 227

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

        # construct model
        self.net = Net(self.num_cls)

        # define optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        if self.gpu_mode:
            self.net = self.net.cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.CE_loss = nn.CrossEntropyLoss()

    def train(self):
        self.train_hist = {}
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        # train
        self.net.train()
        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_start_time = time.time()

            for iB, (img_, label_) in enumerate(self.dataloaders['train']):
                if iB == self.dataloaders['train'].dataset.__len__() // self.batch_size:
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
            utils.loss_plot(self.train_hist, self.result_dir, comment=self.comment)
            self.save()

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Training finished!... save training results")
        print("Total time:", self.train_hist['total_time'][0])
        print("Per epoch time:", self.train_hist['per_epoch_time'])
        self.save()

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        torch.save(self.net.state_dict(), os.path.join(self.save_dir, 'model_' + self.comment + '.pkl'))

        with open(os.path.join(self.save_dir, 'history_' + self.comment + '.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        self.net.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_' + self.comment + '.pkl')))

    def test(self):
        self.net.eval()
        self.load()

        test_loss, correct = 0, 0
        error_cnt = [[0 for x in range(self.num_cls)] for y in range(self.num_cls)]
        with torch.no_grad():
            for img, target in self.dataloaders['test']:
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

        test_loss /= len(self.dataloaders['test'].dataset)

        print('Average loss: %f' % test_loss,
              "Accuracy: %d/%d (%f)" % (correct, len(self.dataloaders['test'].dataset),
                                        100. * correct / len(self.dataloaders['test'].dataset)))

        print('error tracking')
        for i in range(self.num_cls):
            print('class %d:' % i, error_cnt[i],
                  '\taccuracy: %f(%d/%d)' % (error_cnt[i][i] / sum(error_cnt[i]) * 100, error_cnt[i][i], sum(error_cnt[i])))

    def predict(self, img):
        self.net.eval()
        self.load()

        data_transform = transforms.Compose([
                transforms.Resize(self.resl),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor()
        ])

        input = data_transform(img)
        output = self.net(input.expand(1, 3, 227, 227))

        file = open(self.dataroot_dir + 'class.txt')
        classes = file.readlines()[1:]

        sm = torch.nn.Softmax()
        output_list = sm(output).tolist()[0]
        for i in range(len(output_list)):
            form = '{:.5%}'.format(output_list[i])
            print('%-15s' % classes[i].split()[1].strip(), '\t', form, sep='')

        return classes[output.argmax(dim=1, keepdim=True).tolist()[0][0]].split()[1].strip()
