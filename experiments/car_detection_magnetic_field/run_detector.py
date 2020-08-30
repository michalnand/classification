import sys
sys.path.insert(0,'../..')

import libs
import libs_dataset
import numpy

import models.net_0.model as Model0
import models.net_1.model as Model1
import models.net_2.model as Model2
import models.net_3.model as Model3


dataset_path = "/Users/michal/dataset/"
model_path   = "models/net_2/trained/"

confidence_threshold_c1 = 0.38
confidence_threshold_c2 = 0.42
confidence_threshold_c3 = 0.35

window_overlap       = 0.3
sample_skipping      = 8

window_width = 512
#stream       = libs_dataset.StreamMagnetometer(dataset_path  + "/car_detection_2/Meranie_20_06_03-Kinekus/1", width = window_width)
#stream       = libs_dataset.StreamMagnetometer(dataset_path  + "/car_detection_2/Meranie_20_06_03-Kinekus/2", width = window_width)
stream       = libs_dataset.StreamMagnetometer(dataset_path  + "/car_detection_2/Meranie_20_06_01-Lietavska_Lucka/01", width = window_width)



model = Model2.Create(stream.input_shape, stream.output_shape)
model.load(model_path)

cars_stats = numpy.zeros(stream.classes_count)

window_skip = int(window_width*(1.0 - window_overlap))

idx = 0
while idx < stream.get_idx_max():
    time_stamp, data = stream.get_window(idx)

    prediction = model.forward(data.to(model.device))
    prediction = prediction.squeeze(0).detach().to("cpu").numpy()
    prediction_exp = numpy.exp(prediction - numpy.max(prediction))
    prediction_probs = prediction_exp/numpy.sum(prediction_exp)


    if prediction_probs[1] > confidence_threshold_c1:
        #class 1
        cars_stats[1]+= 1
        idx+= window_skip
    elif prediction_probs[2] > confidence_threshold_c2:
        #class 2
        cars_stats[2]+= 1
        idx+= window_skip
    elif prediction_probs[3] > confidence_threshold_c3:
        #class 3
        cars_stats[3]+= 1
        idx+= window_skip
    else:
        #background
        cars_stats[0]+= 1
        idx+= sample_skipping

    done = idx/stream.get_idx_max()
    print(round(done*100.0, 1), stream.stats, cars_stats, numpy.round(prediction_probs, 2))
   

print("\n\n")
print("class_id\t\t", "target_count\t\t", "predicted_count\t\t")
for i in range(len(cars_stats)):
    print(i, "\t\t\t", int(stream.stats[i]), "\t\t\t", int(cars_stats[i]))

print("program done")