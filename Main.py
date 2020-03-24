import math
import datetime
import sys
import numpy as np
import time
from multiprocessing import Process
import multiprocessing
from multiprocessing import Pool
import mmap
import contextlib

def str2float(s):
    return int(s.split('.')[1]) / 1000

class LR:
    def __init__(self, train_file_name, test_file_name, predict_result_file_name):
        self.train_file = train_file_name
        self.predict_file = test_file_name
        self.predict_result_file = predict_result_file_name
        self.max_iters = 37
        self.rate = 3
        self.feats = []
        self.labels = []
        self.feats_test = []
        self.labels_predict = []
        self.param_num = 0
        self.weight = []
        self.offset = 0

    def ReadData(self,n_processing,read_train_lines_set,trainNUM,read_test_lines_set):
        train_data = np.zeros((trainNUM,1001))
        read_lines_count = 0
        f = open(self.train_file, 'r')
        jumpflag = 1
        with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
            m.seek(len(f.readline())*read_train_lines_set*n_processing,0)
            while True:
                line = m.readline().strip()
                if jumpflag == 1:
                    jumpflag = 0
                    continue
                train_data[read_lines_count] = line.split(b',')
                read_lines_count += 1
                if read_lines_count == trainNUM:
                    break

        test_data = np.zeros((read_test_lines_set, 1000))
        read_lines_count = 0
        f = open(self.predict_file, 'r')
        with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
            m.seek(6000 * n_processing * read_test_lines_set, 0)
            while True:
                test_data[read_lines_count] = m.readline().strip().split(b',')
                read_lines_count += 1
                if read_lines_count == read_test_lines_set:
                    break

        return train_data, test_data

    # def ReadData(self,n_processing,read_train_lines_set,trainNUM,read_test_lines_set):
    #     train_data = np.zeros((trainNUM,1001))
    #     read_lines_count = 1
    #     with open(self.train_file,'r') as f:
    #         f.seek(len(f.readline())*read_train_lines_set*n_processing,0)
    #         f.readline()
    #         index = 0
    #         jumpflag = 1
    #         for line in f:
    #             if jumpflag == 1:
    #                 jumpflag = 0
    #                 continue
    #             read_lines_count += 1
    #             train_data[index,:] = list(map(float,line.strip().split(',')))
    #             index += 1
    #             # print(train_data)
    #             if read_lines_count > trainNUM:
    #                 break
    #
    #     test_data = np.zeros((read_test_lines_set,1000))
    #     read_lines_count = 1
    #     with open(self.predict_file,'r') as f:
    #         f.seek(6000*n_processing*read_test_lines_set,0)
    #         lines = f.readlines(read_test_lines_set*6000)
    #         index = 0
    #         for line in lines:
    #             read_lines_count += 1
    #             test_data[index,:] = list(map(float,line.strip().split(',')))
    #             index += 1
    #             if read_lines_count > read_test_lines_set:
    #                 break
    #     return train_data, test_data

    # def ReadData(self,n_processing,read_train_lines_set,trainNUM,read_test_lines_set):
    #     train_data = np.zeros((trainNUM,1001))
    #     read_lines_count = 1
    #     with open(self.train_file,'r') as f:
    #         f.seek(len(f.readline())*read_train_lines_set*n_processing,0)
    #         f.readline()
    #         index = 0
    #         jumpflag = 1
    #         for line in f:
    #             if jumpflag == 1:
    #                 jumpflag = 0
    #                 continue
    #             read_lines_count += 1
    #             train_data[index,:] = list(map(float,line.strip().split(',')))
    #             index += 1
    #             # print(train_data)
    #             if read_lines_count > trainNUM:
    #                 break
    #
    #     test_data = np.zeros((read_test_lines_set,1000))
    #     read_lines_count = 1
    #     with open(self.predict_file,'r') as f:
    #         f.seek(6000*n_processing*read_test_lines_set,0)
    #         index = 0
    #         for line in f:
    #             read_lines_count += 1
    #             test_data[index,:] = list(map(float,line.strip().split(',')))
    #             index += 1
    #             if read_lines_count > read_test_lines_set:
    #                 break
    #     return train_data, test_data

    def loadData(self):
        pro_num = 4
        trf = open(self.train_file)
        trf.seek(0,2)
        read_train_lines_set = int(trf.tell()/6050/pro_num)
        trainNUM = 1500
        trf.close

        tef = open(self.predict_file)
        tef.seek(0, 2)
        read_test_lines_set = int(tef.tell() / 6000 / pro_num)
        tef.close

        pool = Pool(processes = pro_num)
        job_result = []
        for i in range(pro_num):
            res = pool.apply_async(self.ReadData, (i, read_train_lines_set,trainNUM,read_test_lines_set))
            job_result.append(res)
        pool.close()
        pool.join()

        for tmp in job_result:
            temp_train, temp_test = tmp.get()
            self.feats.extend(temp_train)
            self.labels.extend(temp_train)
            self.feats_test.extend(temp_test)
        self.feats = np.array(self.feats)
        self.labels = self.feats[:,-1]
        self.feats = self.feats[:,:-1]

    def savePredictResult(self):
        f = open(self.predict_result_file, 'w')
        for i in range(len(self.labels_predict)):
            f.write(str(self.labels_predict[i])+"\n")
        f.close()

    def printInfo(self):
        print(self.train_file)
        print(self.predict_file)
        print(self.predict_result_file)
        # print(self.feats)
        print('训练样本数：',len(self.labels))
        print('测试样本数', len(self.labels_predict))
        print('特征维度：',len(self.feats_test[0]))

    def initParams(self):
        self.weight = np.zeros((self.param_num,), dtype=np.float)

    def compute(self, recNum, param_num, feats, w):
        return 1 / (1 + np.exp(-np.dot(feats, w)))

    def error_rate(self, recNum, label, preval):
        return np.power(label - preval, 2).sum()

    def predict(self,data):
        preval = self.compute(len(data),
                              self.param_num, data, self.weight)
        labels_predict = (preval + 0.5).astype(np.int)
        return labels_predict

    def train(self):
        # time3 = time.time()
        self.loadData()
        # time4 = time.time()
        # print('read Txt cost time: ',time4-time3)
        # time5 = time.time()
        recNum = len(self.feats)
        self.param_num = len(self.feats[0])
        self.initParams()
        ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S,f'
        for i in range(self.max_iters):
            preval = self.compute(recNum, self.param_num,
                                  self.feats, self.weight)
            sum_err = self.error_rate(recNum, self.labels, preval)
            # if i%1 == 0:
            #     print("Iters:" + str(i) + " error:" + str(sum_err))
            #     theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
            #     print(theTime)
            err = self.labels - preval
            self.miniSGD(err=err)
            self.rate = self.rate*0.88

        # time6 = time.time()
        # print('train cost time: ',time6-time5)

    def miniSGD(self,err):
        minibatch_size = 1500
        down_pos = self.offset
        up_pos = self.offset+minibatch_size
        if up_pos > len(self.feats):
            down_pos = 0
            up_pos = down_pos + minibatch_size
        minibatch_feats = self.feats[range(down_pos,up_pos),:]
        minibatch_err = err[range(down_pos,up_pos)]
        delt_w = np.dot(minibatch_feats.T,minibatch_err)/minibatch_size
        self.weight += self.rate*delt_w
        if up_pos == len(self.feats):
            self.offset = 0
        else:
            self.offset = up_pos

def print_help_and_exit():
    print("usage:python3 main.py train_data.txt test_data.txt predict.txt [debug]")
    sys.exit(-1)


def parse_args():
    debug = False
    if len(sys.argv) == 2:
        if sys.argv[1] == 'debug':
            print("test mode")
            debug = True
        else:
            print_help_and_exit()
    return debug


if __name__ == "__main__":
    # StartTime = time.time()
    debug = parse_args()
    train_file =  "/data/train_data.txt"
    test_file = "/data/test_data.txt"
    predict_file = "/projects/student/result.txt"
    lr = LR(train_file, test_file, predict_file)

    lr.train()
    # time3 = time.time()
    lr.labels_predict = lr.predict(lr.feats_test)
    # time4 = time.time()
    # print('predict cost time: ',time4-time3)
    # time5 = time.time()
    lr.savePredictResult()
    # time6 = time.time()
    # print('save Txt cost time: ',time6-time5)

    if debug:
        answer_file ="./projects/student/answer.txt"
        f_a = open(answer_file, 'r')
        f_p = open(predict_file, 'r')
        a = []
        p = []
        lines = f_a.readlines()
        for line in lines:
            a.append(int(float(line.strip())))
        f_a.close()

        lines = f_p.readlines()
        for line in lines:
            p.append(int(float(line.strip())))
        f_p.close()

        errline = 0
        for i in range(len(a)):
            if a[i] != p[i]:
                errline += 1

        accuracy = (len(a)-errline)/len(a)
        print("accuracy:%f" %(accuracy))
        EndTime = time.time()

        print('Debug Time: ',EndTime-StartTime)