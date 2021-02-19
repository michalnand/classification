import torch
import numpy


class MetricsSegmentation:
    def __init__(self, output_shape):
        self.classes_count  = output_shape[0]

        self._loss_training = []
        self._loss_testing  = []

        self._iou_training  = []
        self._iou_testing   = []

        self._class_iou_training     = []
        self._class_iou_testing      = []
        

    def loss_training(self, target_t, predicted_t):
        return ((target_t.detach() - predicted_t)**2).mean()

    def loss_testing(self, target_t, predicted_t):
        return ((target_t.detach() - predicted_t)**2).mean()

    def add_training(self, target_t, predicted_t):
        loss = self.loss_training(target_t, predicted_t)

        target_ids      = target_t.argmax(axis=1)
        predicted_ids   = predicted_t.argmax(axis=1)

        iou             = self._iou(target_ids, predicted_ids)
        class_iou       = self._class_iou(target_ids, predicted_ids)

        self._loss_training.append(loss.detach().to("cpu").numpy())
        self._iou_training.append(iou)
        self._class_iou_training.append(class_iou)


    def add_testing(self, target_t, predicted_t):
        loss = self.loss_testing(target_t, predicted_t)

        target_ids      = target_t.argmax(axis=1)
        predicted_ids   = predicted_t.argmax(axis=1)

        iou             = self._iou(target_ids, predicted_ids)
        class_iou       = self._class_iou(target_ids, predicted_ids)

        self._loss_testing.append(loss.detach().to("cpu").numpy())
        self._iou_testing.append(iou)
        self._class_iou_testing.append(class_iou)

    def compute(self):
        self.loss_training_mean     = numpy.array(self._loss_training).mean()
        self.loss_training_std      = numpy.array(self._loss_training).std()
        self.loss_testing_mean      = numpy.array(self._loss_testing).mean()
        self.loss_testing_std       = numpy.array(self._loss_testing).std()        

        self.iou_training_mean     = numpy.array(self._iou_training).mean()
        self.iou_training_std      = numpy.array(self._iou_training).std()
        self.iou_testing_mean      = numpy.array(self._iou_testing).mean()
        self.iou_testing_std       = numpy.array(self._iou_testing).std()

        self.class_iou_training     = numpy.array(self._class_iou_training).mean(axis=0)
        self.class_iou_testing      = numpy.array(self._class_iou_testing).mean(axis=0)

    def get_score(self):
        return self.iou_testing_mean

    def get_short(self):

        result = ""
        result+= str(self.loss_training_mean) + " "
        result+= str(self.loss_training_std) + " "
        result+= str(self.iou_training_mean) + " "
        result+= str(self.iou_training_std) + " "

        result+= str(self.loss_testing_mean) + " "
        result+= str(self.loss_testing_std) + " "
        result+= str(self.iou_testing_mean) + " "
        result+= str(self.iou_testing_std) + " "

        return result

    def get_full(self):

        result = ""
        result+= "TRAINING result\n\n"
        result+= "loss_mean = " + str(self.loss_training_mean) + "\n"
        result+= "loss_std  = " + str(self.loss_training_std) + "\n"
        result+= "iou_mean = " + str(self.iou_training_mean) + "\n"
        result+= "iou_std  = " + str(self.iou_training_std) + "\n"

        result+= "class_iou  = \n"
        for i in range(self.classes_count):
            result+= "{:>12}".format(round(self.class_iou_training[i], 5))

        result+= "\n\n\n\n"

        result+= "TESTING result\n\n"
        result+= "loss_mean = " + str(self.loss_testing_mean) + "\n"
        result+= "loss_std  = " + str(self.loss_testing_std) + "\n"
        result+= "iou_mean = " + str(self.iou_testing_mean) + "\n"
        result+= "iou_std  = " + str(self.iou_testing_std) + "\n"

        result+= "class_iou  = \n"
        for i in range(self.classes_count):
            result+= "{:>12}".format(round(self.class_iou_testing[i], 5))

        return result

    def _iou(self, target_ids_t, predicted_ids_t):
        intersection    = (target_ids_t == predicted_ids_t).sum().detach().to("cpu").numpy()
        union           = numpy.prod(target_ids_t.shape)
        
        return intersection*1.0/(union + 1.0)


    def _class_iou(self, target_t, predicted_t):

        result = numpy.zeros(self.classes_count)

        for class_id in range(self.classes_count):
            target_ids_fil_t    = (target_t == class_id)
            predicted_ids_fil_t = (predicted_t == class_id)

            intersection        = torch.logical_and(target_ids_fil_t, predicted_ids_fil_t).sum().detach().to("cpu").numpy()
            union               = torch.logical_or(target_ids_fil_t, predicted_ids_fil_t).sum().detach().to("cpu").numpy()

            result[class_id] = intersection*1.0/(union + 1.0)

        return result