from socket import *
import sys, os, argparse
from PIL import Image
from CNN import CNN
from foodRecipe import get_foodInfo
import shutil

HOST = ''
PORT = 16161
BUFSIZE = 1024
ADDR = (HOST, PORT)
CLIENT_NUM = 5
COMMENT = 'data_clean_70epoch'
DATAROOT_DIR = './data_clean/'
FOLD_NUM = -1

def main():
    # 모델 불러오기
    opts = parse_args()
    model = CNN(opts)

    # 소켓 생성
    serverSocket = socket(AF_INET, SOCK_STREAM)

    # 소켓 주소 정보 할당
    serverSocket.bind(ADDR)
    print('bind')

    # 연결 수신 대기 상태
    serverSocket.listen(CLIENT_NUM)
    print('listen')

    while True:
        print('waiting...')
        try:
            # 연결 수락
            connectionSocket, addr_info = serverSocket.accept()
            print('accept')
            print('--client information--')
            print(connectionSocket)

            recvImg(connectionSocket)

            pred_results = model.predict(img=Image.open('./img.jpg'), fold_num=opts.fold_num)

            sendRecipe(connectionSocket, pred_results[0][0], pred_results[0][1])
            print("---------first recipe send done---------")

            print(connectionSocket)
            response = connectionSocket.recv(10).decode()
            print(response);
            if 'Y' in response:
                print("pred correct1")
                saveImg('./received_data', pred_results[0][0])
            elif 'N' in response:
                print("pred wrong1")
                sendRecipe(connectionSocket, pred_results[1][0], pred_results[1][1])
                print("---------second recipe send done---------")
                response = connectionSocket.recv(10).decode()
                if 'Y' in response:
                    print("pred correct2")
                    saveImg('./received_data', pred_results[1][0])
                elif 'N' in response:
                    print("prew wrong2")

            connectionSocket.shutdown(SHUT_RDWR)
            connectionSocket.close()
            print('connection closed')

        except KeyboardInterrupt:
            sys.exit(0)


def recvImg(connectionSocket):
    # 헤더 정보 읽기
    fileSize = int(connectionSocket.recv(32).decode().replace('\0', ''))

    # 클라이언트로부터 파일을 가져옴
    img = open("./img.jpg", 'wb')
    img_data = connectionSocket.recv(BUFSIZE)
    data = img_data
    print("receiving Img...")
    while len(data) < fileSize:
        img_data = connectionSocket.recv(BUFSIZE)
        data += img_data
    print("finish img recv", len(data))
    img.write(data)
    img.close()


def sendRecipe(connectionSocket, idFood, predVal):
    # DB에 질의
    query_result = get_foodInfo(idFood)

    # 레시피 정보를 클라이언트에게 보냄
    connectionSocket.send((str(idFood) + '\r\n').encode())
    connectionSocket.send("food id\r\n".encode())

    form = '{:.5%}'.format(predVal)
    connectionSocket.send((form + '\r\n').encode())
    connectionSocket.send("predict percentage\r\n".encode())

    section_ko = ["food name\r\n", "food ingredients\r\n", "food preparation\r\n", "food cooking\r\n"]
    recipe = query_result[1][1:]
    for i in range(len(recipe)):
        connectionSocket.send((recipe[i] + '\r\n').encode("CP949"))
        connectionSocket.send(section_ko[i].encode())

    section_en = ["food name\r\n", "food krName\r\n", "food ingredients\r\n", "food preparation\r\n",
                  "food cooking\r\n"]
    recipe_en = query_result[2][1:]
    connectionSocket.send("recipe_en\r\n".encode())
    for i in range(len(recipe_en)):
        connectionSocket.send((recipe_en[i] + '\r\n').encode("CP949"))
        connectionSocket.send(section_en[i].encode())
    connectionSocket.send("recipe_done\r\n".encode())


def saveImg(save_root_dir, food_id):
    # food_id에 해당하는 dir에 이미지 저장
    save_dir = save_root_dir + '/' + str(food_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_cnt = len(os.walk(save_dir).__next__()[2])
    shutil.move('./img.jpg', save_dir + '/' + str(food_id) + '_' + '{:04}'.format(file_cnt) + '.jpg')


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

    parser.add_argument('--dataroot_dir', type=str, default=DATAROOT_DIR, help='Root path of data')
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    parser.add_argument('--sample_num', type=int, default=64, help='The number of samples to test')
    parser.add_argument('--save_dir', type=str, default='./model/', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='./results/', help='Directory name to save the result')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=str_to_bool, default='False')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of threads')
    parser.add_argument('--comment', type=str, default=COMMENT, help='Comment to pyt on model_name')
    parser.add_argument('--type', type=str, default='pred', help='train or test or pred')
    parser.add_argument('--fold_num', type=int, default=FOLD_NUM, help='fold number')

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


main()

