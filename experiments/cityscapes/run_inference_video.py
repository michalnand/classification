import numpy as np
import cv2

from segmentation_inference import *
import models.model_0.model as Model0

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/Users/michal/Movies/innsbruck.mp4")

si = SegmentationInference(Model0, "models/model_0/trained/")

fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (512, 256)) 


cnt = 0
while(True):
    ret, frame = cap.read()

    frame = cv2.resize(frame, (512, 256), interpolation = cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if cnt%10 == 0:
        prediction_np, mask, result = si.process(frame)

        result = (result*255).astype(numpy.uint8)

        im_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', im_bgr)

        out.write(im_bgr)

    cnt+= 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
