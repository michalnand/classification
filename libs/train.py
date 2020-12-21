import numpy
import torch
import time

from .loss import *
from .confussion_matrix import *

class Train:
    def __init__(self, dataset, Model, batch_size = 64, learning_rates = [0.0001], weight_decay = 0.001, loss = LossMSE):

        self.dataset        = dataset
        self.model          = Model.Create(self.dataset.input_shape, self.dataset.output_shape)

        self.batch_size       = batch_size
        self.learning_rates   = learning_rates
        self.weight_decay     = weight_decay

        self.loss             = loss

    def step_epochs(self, epoch_count, log_path = "./"):
        accuracy_best       = -1.0
        epoch_time_filtered = -1.0

        f_training_log  = open(log_path + "/result/training.log","w+")

        for epoch in range(epoch_count):
            learning_rate = self.learning_rates[epoch_count%len(self.learning_rates)]
            training_confussion_matrix, testing_confussion_matrix, training_loss, testing_loss, epoch_time = self.step_epoch(learning_rate, epoch, epoch_count)

            if epoch_time_filtered < 0.0:
                epoch_time_filtered = epoch_time
            else:
                epoch_time_filtered = 0.9*epoch_time_filtered + 0.1*epoch_time

            eta_time    = (epoch_count - epoch)*(epoch_time_filtered/3600.0)

            training_accuracy   = training_confussion_matrix.accuracy
            testing_accuracy    = testing_confussion_matrix.accuracy

            training_loss_mean  = numpy.mean(training_loss)
            testing_loss_mean   = numpy.mean(testing_loss)

            training_loss_std   = numpy.std(training_loss)
            testing_loss_std    = numpy.std(testing_loss)

            log_str = ""
            log_str+= str(epoch) + " "
            log_str+= str(training_accuracy) + " "
            log_str+= str(testing_accuracy) + " "
            log_str+= str(training_loss_mean) + " "
            log_str+= str(testing_loss_mean) + " "
            log_str+= str(training_loss_std) + " "
            log_str+= str(testing_loss_std) + " "
            log_str+= str(round(eta_time, 2)) + " "
            log_str+= "\n"

            print(log_str)
            f_training_log.write(log_str)
            f_training_log.flush()

            if testing_accuracy > accuracy_best:
                self.model.save(log_path + "/trained/")

                accuracy_best = testing_accuracy

                log_str = ""
                log_str+= "new best net in " + str(epoch) + "\n"
                log_str+= "TRAINING result\n"
                log_str+= training_confussion_matrix.get_result() + "\n\n"
                log_str+= "TESTING result\n"
                log_str+= testing_confussion_matrix.get_result() + "\n\n"

                print("\n\n\n")
                print("=================================================")
                print(log_str)
                
                f_best_log = open(log_path + "/result/best.log","w+")
                f_best_log.write(log_str)
                f_best_log.close()

        f_training_log.close()
    

    def step_epoch(self, learning_rate, epoch, epoch_count):

        time_start = time.time()

        if hasattr(self.model, 'epoch_start'):
            self.model.epoch_start(epoch, epoch_count)

        optimizer  = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=learning_rate*self.weight_decay)  

        batch_count = (self.dataset.get_training_count() + self.batch_size) // self.batch_size

        training_confussion_matrix = ConfussionMatrix(self.dataset.classes_count)
        
        training_loss = []
        for batch_id in range(batch_count):
            training_x, training_y = self.dataset.get_training_batch(self.batch_size)

            training_x = training_x.to(self.model.device)
            training_y = training_y.to(self.model.device)

            predicted_y = self.model.forward(training_x)

            loss  = self.loss(training_y, predicted_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_confussion_matrix.add_batch(training_y.detach().to("cpu").numpy(), predicted_y.detach().to("cpu").numpy())

            training_loss.append(loss.detach().to("cpu").numpy())

        training_confussion_matrix.compute()
 
        batch_count = (self.dataset.get_testing_count()+self.batch_size) // self.batch_size
        testing_confussion_matrix = ConfussionMatrix(self.dataset.classes_count)

        testing_loss = []
        for batch_id in range(batch_count):
            testing_x, testing_y = self.dataset.get_testing_batch(self.batch_size)

            testing_x = testing_x.to(self.model.device)
            testing_y = testing_y.to(self.model.device)

            predicted_y = self.model.forward(testing_x)

            error = (testing_y - predicted_y)**2
            loss  = error.mean()

            testing_confussion_matrix.add_batch(testing_y.detach().to("cpu").numpy(), predicted_y.detach().to("cpu").numpy())
            testing_loss.append(loss.detach().to("cpu").numpy())

        testing_confussion_matrix.compute()

        time_stop = time.time()

        epoch_time = time_stop - time_start

        return training_confussion_matrix, testing_confussion_matrix, training_loss, testing_loss, epoch_time
  