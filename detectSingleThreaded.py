'''
Bibtex citation:

@article{Dibia2017,
  author = {Victor, Dibia},
  title = {HandTrack: A Library For Prototyping Real-time Hand TrackingInterfaces using Convolutional Neural Networks},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/victordibia/handtracking/tree/master/docs/handtrack.pdf}, 
}

'''

from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np

#import send_sms #Script to send SMS if alarm

detection_graph, sess = detector_utils.load_inference_graph()

def send_sms():
    from twilio.rest import Client

    # Your Account Sid and Auth Token from twilio.com/console
    # DANGER! This is insecure. See http://twil.io/secure
    account_sid = 'AC468e8e00e4f9472b4851cd77e533cd91'
    auth_token = '8b9348526a36abd1852dc554ebbd9e05'
    client = Client(account_sid, auth_token)

    message = client.messages \
            .create(
                body='Suspicious activity has been reported!',
                from_='+13437002917',
                to='+16476086035'
            )
   
    print(message.sid)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.5,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=14.95,
        help='Show FPS on detection/display visualization')
    parser.add_argument( #TO DO: video source file.
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=1920,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=1088,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-wdtcrp',
        '--width_cropped',
        dest='width_cropped',
        type=int,
        default=1920,
        help='Width of the frames on which inference is to be done.')
    parser.add_argument(
        '-htcrp',
        '--height_cropped',
        dest='height_cropped',
        type=int,
        default=1088,
        help='Height of the frames on which inference is to be done.')
    parser.add_argument(
        '-crnx',
        '--corner_x',
        dest='top_left_x',
        type=int,
        default=0,
        help='top left corner x (w) of the frames on which inference is to be done.')
    parser.add_argument(
        '-crny',
        '--corner_y',
        dest='top_left_y',
        type=int,
        default=0,
        help='top left corner y (h) of the frames on which inference is to be done.')
    parser.add_argument( #USAGE: 0=no video, 1=infer full res video, 2=infer cropped video + display full res video
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=2,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument( #TO DO: TBD
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')

    #SAMPLE CALL (for video clip #9):
    # python3 detectSingleThreaded.py -crnx 700 -crny 480 -wdtcrp 500 -htcrp 400 -src ~/Documents/final1.mp4
    # python3 detectSingleThreaded.py -crnx 760 -crny 0 -wdtcrp 470 -htcrp 1088 -src ~/Documents/final2.mp4
    # python3 detectSingleThreaded.py -crnx 750 -crny 0 -wdtcrp 450 -htcrp 1088 -src ~/Documents/final3.mp4
    
    # max number of hands we want to detect/track
    num_hands_detect = 1

    ALARM_REGION_TOP_FRAC = 0.2 #fractional region of video in which we should sound an alarm

    NUM_ALARMS = 0


    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)


    print('args.top_left_x ='); print(args.top_left_x) #f'args.top_left_x = {args.top_left_x}')
    print('args.top_left_y ='); print(args.top_left_y)

    print('args.height_cropped ='); print(args.height_cropped)
    print('args.width_cropped ='); print(args.width_cropped)

    while cap.isOpened():
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        
        print('image_np.shape=');print(image_np.shape)
        print('asserted?=');print((args.height,args.width,3))

        #Crop to frame
        if (args.display == 2):
            assert image_np.shape == (args.height,args.width,3), 'Image not a 3D array'

            crp_image_np = image_np[args.top_left_y:args.top_left_y+args.height_cropped, 
                                    args.top_left_x:args.top_left_x+args.height_cropped, :]

        try:
            crp_image_np = cv2.cvtColor(crp_image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(crp_image_np,
                                                      detection_graph, sess)

        print(boxes.shape)
        assert boxes.shape == (100,4), 'Not 100x4 array'

        uncrp_boxes=np.zeros((2,4))

        #Add bounding boxes
        for i in range(num_hands_detect):
            #First get corner pixels
            print('left decimal='); print(boxes[i][1])
            print('right decimal='); print(boxes[i][3])
            print('top decimal='); print(boxes[i][0])
            print('bottom decimal='); print(boxes[i][2])
            (left, right, top, bottom) = (boxes[i][1] * args.width_cropped, boxes[i][3] * args.width_cropped,
                                          boxes[i][0] * args.height_cropped, boxes[i][2] * args.height_cropped)

            print('Coords inside=');print((left, right, top, bottom));print('\n')
            
            #Add offsets and convert back to box objects
            (left_w_offs, right_w_offs, top_w_offs, bottom_w_offs) = (left + args.top_left_x, right + args.top_left_x, top + args.top_left_y, bottom + args.top_left_y)

            print('im_height=');print(im_height)

            (uncrp_boxes[i][1],uncrp_boxes[i][3],uncrp_boxes[i][0],uncrp_boxes[i][2]) = (left_w_offs/im_width,right_w_offs/im_width,top_w_offs/im_height,bottom_w_offs/im_height)
            print('Decimals outside='); print((uncrp_boxes[i][1],uncrp_boxes[i][3],uncrp_boxes[i][0],uncrp_boxes[i][2]))

            assert (not(uncrp_boxes[i][0] > 1)) & (not(uncrp_boxes[i][1] > 1)) & (not(uncrp_boxes[i][2] > 1)) & (not(uncrp_boxes[i][3] > 1)), 'Improper uncropped image boxes'

        # draw bounding boxes on frame
        detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                         scores, uncrp_boxes, im_width, im_height,
                                         image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            #### LOGIC FOR ALARM DETECTION: top left corner y within ALARM_REGION_TOP_FRAC
            alarm_str = 'False'

            if(boxes[0][0]<ALARM_REGION_TOP_FRAC):
                if(NUM_ALARMS==5): #ignore the first few (error)
                    alarm_str = 'True'
                    #Send SMS if alarm
                    send_sms()
                NUM_ALARMS = NUM_ALARMS +1 

            detector_utils.draw_str_on_image("Alarm : " + alarm_str, image_np, round(0.8*im_width), round(0.8*im_height))


            print('Alarm = '); print(alarm_str)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))