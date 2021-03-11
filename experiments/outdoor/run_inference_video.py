import cv2
import time

from segmentation_inference import *
import models.model_0.model as Model

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/Users/michal/Movies/segmentation/park.mp4")
#cap = cv2.VideoCapture("/Users/michal/Movies/segmentation/street_01.mp4")
#cap = cv2.VideoCapture("/Users/michal/Movies/segmentation/street_03.mp4")
#cap = cv2.VideoCapture("/Users/michal/Movies/segmentation/street_04.mp4")




show_video = True
save_video = False

height  = 256 #480
width   = 512 #640


si = SegmentationInference(Model,  "models/model_0/trained/", 5, height, width)


if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    writer = cv2.VideoWriter('segmentation_output_2.avi', fourcc, 5.0, (width, height)) 


fps_smooth  = 10.0
frame_skip  = 1
next_frame  = 0
cnt         = 0

def print_video(image, text):
    x = cv2.putText(image,text,(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,lineType=cv2.LINE_AA)

while(True): 

    ts = time.time()
    ret, frame = cap.read()

    if ret == False:
        break

    frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    te = time.time()

    time_preprocessing = te - ts

    if cnt > next_frame:
        ts = time.time()
        prediction_np, mask, result = si.process(frame)
        te  = time.time()

        time_model = te - ts

        ts = time.time()

        #show video
        result = (result*255).astype(numpy.uint8)

        text  = "fps= " + str(round(fps_smooth, 1))

        im_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        print_video(im_bgr, text)

        if show_video:
            cv2.imshow('frame', im_bgr)

        if save_video:
            writer.write(im_bgr)

        te = time.time()

        time_postprocessing = te - ts

        #compute FPS and frame skipping
        time_total = time_preprocessing + time_model + time_postprocessing

        fps_smooth = 0.8*fps_smooth + 0.2*(1.0/time_total)
        frame_skip = 25/fps_smooth
        frame_skip = int(numpy.clip(frame_skip, 1, 500))

        next_frame = cnt + frame_skip

        p_preprocessing  = round(100.0*time_preprocessing/time_total, 3)
        p_model          = round(100.0*time_model/time_total, 3)
        p_postprocessing = round(100.0*time_postprocessing/time_total, 3)

        print("fps                 = "  , round(fps_smooth, 2))
        print("skipped frames      = "  , frame_skip)
        print("preprocessing       = "  , p_preprocessing, "[%]")
        print("model               = "  , p_model, "[%]")
        print("postprocessing      = "  , p_postprocessing, "[%]")
        print("\n\n\n")

    cnt+= 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
