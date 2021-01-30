import numpy
import torch
import time

class Train:
    def __init__(self, dataset, Model, Metrics, batch_size = 64, learning_rates = [0.0001], weight_decay = 0.001):

        self.dataset            = dataset
        self.model              = Model.Create(self.dataset.input_shape, self.dataset.output_shape)

        self.batch_size         = batch_size
        self.learning_rates     = learning_rates
        self.weight_decay       = weight_decay

        self.Metrics            = Metrics

    def step_epochs(self, epoch_count, log_path = "./"):
        accuracy_best       = -1.0
        epoch_time_filtered = -1.0

        f_training_log  = open(log_path + "/result/training.log","w+")

        best_score = 0.0

        for epoch in range(epoch_count):
            learning_rate = self.learning_rates[epoch_count%len(self.learning_rates)]
            metrics, epoch_time = self.step_epoch(learning_rate, epoch, epoch_count)

            if epoch_time_filtered < 0.0:
                epoch_time_filtered = epoch_time
            else:
                epoch_time_filtered = 0.9*epoch_time_filtered + 0.1*epoch_time

            eta_time    = (epoch_count - epoch)*(epoch_time_filtered/3600.0)

            log_str = ""
            log_str+= str(epoch) + " "
            log_str+= metrics.get_short() + " "
            log_str+= str(round(eta_time, 2)) + " "
            log_str+= "\n"

            print(log_str)
            f_training_log.write(log_str)
            f_training_log.flush()

            if metrics.get_score() > best_score:
                self.model.save(log_path + "/trained/")

                best_score = metrics.get_score()

                log_str = "best model\n"
                log_str+= "epoch = " + str(epoch) + "\n\n"
                log_str+= metrics.get_full()
 
                f_best_log = open(log_path + "/result/best.log","w+")
                f_best_log.write(log_str)
                f_best_log.close()

                print(log_str, "\n\n\n")

        f_training_log.close()
    

    def step_epoch(self, learning_rate, epoch, epoch_count):

        time_start = time.time()

        if hasattr(self.model, 'epoch_start'):
            self.model.epoch_start(epoch, epoch_count)

        optimizer   = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=learning_rate*self.weight_decay)  
        batch_count = (self.dataset.get_training_count() + self.batch_size) // self.batch_size

        metrics            = self.Metrics(self.dataset.output_shape)

        for batch_id in range(batch_count):
            training_x, training_y = self.dataset.get_training_batch(self.batch_size)

            training_x = training_x.to(self.model.device)
            training_y = training_y.to(self.model.device)

            predicted_y = self.model.forward(training_x)

            loss  = metrics.loss_training(training_y, predicted_y)
            metrics.add_training(training_y, predicted_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
        batch_count = (self.dataset.get_testing_count()+self.batch_size) // self.batch_size

        testing_loss = []
        for batch_id in range(batch_count):
            testing_x, testing_y = self.dataset.get_testing_batch(self.batch_size)

            testing_x = testing_x.to(self.model.device)
            testing_y = testing_y.to(self.model.device)

            predicted_y = self.model.forward(testing_x)

            error = (testing_y - predicted_y)**2
            loss  = metrics.loss_testing(training_y, predicted_y)

            metrics.add_testing(testing_y, predicted_y)

        metrics.compute()

        time_stop   = time.time()
        epoch_time  = time_stop - time_start

        return metrics, epoch_time
  