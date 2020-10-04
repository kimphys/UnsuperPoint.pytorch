class args():

    # hyperparameters
    a_usp = 1
    a_pos = 1
    a_scr = 2
    a_xy = 100
    a_desc = 0.001
    a_decorr = 0.03
    lambda_d = 250
    margin_p = 1
    margin_n = 0.2

    # training args
    epochs = 10 # "number of training epochs, default is 2"
    save_per_epoch = 1
    batch_size = 6 # "batch size for training, default is 4"
    dataset = "/home/kim/ai/dataset/coco_train.txt"
    HEIGHT = 240
    WIDTH = 240
    lr = 1e-4 #"learning rate, default is 0.001"	
    # resume = "models/rgb.pt" # if you have, please put the path of the model like "./models/densefuse_gray.model"
    resume = None
    save_model_dir = "./models/" #"path to folder where trained model with checkpoints will be saved."

    # For GPU training
    world_size = -1
    rank = -1
    dist_backend = 'nccl'
    gpu = None
    multiprocessing_distributed = False
    distributed = None

    # For testing
    test_save_dir = "./"
    test_img = "./test_rgb.txt"
