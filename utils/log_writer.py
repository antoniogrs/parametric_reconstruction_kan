import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd


try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None

'''
regex: ^(?<group>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:\\\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})?.*?\\(?<loss_type>train_loss|val_loss)$
'''
    
class LogWriter:
    previous_epoch = -1
    # def __init__(self, is_train, path = '', epochs=1, base_folder = 'runs', other_info = None):
    #     if is_train and SummaryWriter is not None:
    #         current_datetime = datetime.now()
    #         date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    #         if path == '':
    #             # print(date_time_string)
    #             final_path = date_time_string + '_' + str(epochs) + '_epochs'
    #             if other_info is not None:
    #                 final_path += '_' + other_info
    #             self.display = SummaryWriter(logdir=os.path.join(base_folder, final_path))
    #         else:
    #             final_path = date_time_string
    #             if other_info is not None:
    #                 final_path += '_' + other_info
    #             self.display = SummaryWriter(logdir=os.path.join(path, final_path))
    #         print(self.display.logdir)
    #     else:
    #         self.display = None
            
    def __init__(self, base_folder, loss_folder_name):
        if SummaryWriter is not None:
            self.loss_folder_name = loss_folder_name
            current_datetime = datetime.now()
            date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            final_path = date_time_string 
            # print(os.path.join(base_folder, final_path))
            # raise TypeError
            self.display = SummaryWriter(logdir=os.path.join(base_folder, final_path))
            print(self.display.logdir)
        else:
            self.display = None
            
    def write_params(self, table_name, params_dict):
        table_str = ''
        for key, value in params_dict.items():
            table_str += f'{key}: {value}  \n'
        self.display.add_text(tag=table_name, text_string=table_str)
        
    
    def print_train_loss(self, epoch, i, loss):
        """ 
        prints train loss to terminal / file 
        """
        
        message = '(time: %s, epoch: %d, iters: %d) loss: %.3f ' \
                  % (time.strftime("%X %x"), epoch, i, loss.item())
        if epoch == self.previous_epoch:
            print(message, end='\r')
        else:
            print(message)
        self.previous_epoch = epoch
        # with open(self.log_name, "a") as log_file:
        #     log_file.write('%s\n' % message)

    def plot_train_loss(self, loss, iters):
        # iters = i + epoch * n
        if self.display:
            # self.display.add_scalar('data/train_loss', loss, iters)
            self.display.add_scalars('data/' + self.loss_folder_name, {'train_loss': loss}, iters)

    def print_eval_loss(self, epoch, i, loss):
        """ 
        prints eval loss to terminal / file 
        """
        epoch = 0 if not epoch else epoch
        message = 'eval (time: %s, epoch: %d, iters: %d) loss: %.3f ' \
                  % (time.strftime("%X %x"), epoch, i, loss.item())
        print(message)
        # with open(self.log_name, "a") as log_file:
        #     log_file.write('%s\n' % message)


    def plot_eval_loss(self, loss, epoch):
        if self.display:
            # self.display.add_scalar('data/eval_loss', loss, epoch)
            self.display.add_scalars('data/' + self.loss_folder_name, {'val_loss': loss}, epoch)

    def add_further_info(self, text):
        if text != '':
            self.display.add_text('further_info', text)    

    def close(self):
        if self.display is not None:
            self.display.close()
    
# LogWriter(is_train=True, is_heterodata=True, epochs=50)