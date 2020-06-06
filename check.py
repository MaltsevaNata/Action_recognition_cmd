from ActionRecognitionClass import ActionRecognition


paths_to_classes = ['/home/natalia/openpose/examples/media/check_act_rec/Move_scaner/', '/home/natalia/openpose/examples/media/check_act_rec/Tune_angle/',
                    '/home/natalia/openpose/examples/media/check_act_rec/Tune_height/']

act_rec = ActionRecognition(PCA=True)
#act_rec.train(paths_to_classes, ['Move_scaner', 'Tune_angle', 'Tune_height'], 'svm')
act_rec.load_trained('svm', actions_names=['Move_scaner', 'Tune_angle', 'Tune_height'])
#act_rec.predict('/home/natalia/openpose/examples/media/check_act_rec/Tune_height/Tune_height2.mp4')
act_rec.real_time_prediction('/home/natalia/openpose/examples/media/check_act_rec/Tune_height/Tune_height2.mp4')