import os
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from src.CV_IO_utils import read_imgs_dir
from src.CV_transform_utils import apply_transformer
from src.CV_transform_utils import resize_img, normalize_img
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


# Run mode: (transfer learning -> vgg19,resnet50 )
#modelName = "vgg19"  # try: "vgg19", "resnet50"
trainModel = False   # As we are using pre-trained model, it will be false. 
parallel = False  # use multicore processing, if gpu available

def main_model(img_list,img_query,model_name,knn_admin):
    print(model_name)
    modelName = model_name  # try: "vgg19", "resnet50"
    print('number of k:',knn_admin)
    dataTrainDir = os.path.join(os.getcwd(), "data", "product_images")
    dataTestDir = os.path.join(os.getcwd(), "data", "product_images")
    outDir = os.path.join(os.getcwd(), "output", modelName) # Creates a output folder to save our retrieved images.
    
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # Read images from the given directories.
    extensions = [".jpg", ".jpeg"]
    print("Reading train images from '{}'...".format(dataTrainDir))
    imgs_train = read_imgs_dir(dataTrainDir, extensions,img_list, parallel=parallel) #imported from src.CV_IO_utils
    print('*!@#*len of training images:', len(imgs_train))
    print("Reading test images from '{}'...".format(dataTestDir))
    imgs_test = read_imgs_dir(dataTestDir, extensions, img_query, parallel=parallel)
    print('*!@#*len of test images:', len(imgs_test))
    shape_img = imgs_train[0].shape
    print("Image shape = {}".format(shape_img))

    # based on the selected model, the function will run. 
    if modelName in ["vgg19"]:

        # Load pre-trained VGG19 model + higher level layers
        # have are using the imagenet weights and the classification layer is removed.
        # i.e include_top=False.
        print("Loading VGG19 pre-trained model...")
        model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                            input_shape=shape_img)
        model.summary()

        shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
        input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
        output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
        #n_epochs = None as it is pre-trained.
        n_epochs = None
    elif modelName in ["resnet50"]:
    
        print("Loading the Resnet50 pre-trained model...")
        model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, 
                                                input_shape=shape_img)
    
        model.summary()

        shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
        input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
        output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
        #n_epochs = None as it is pre-trained.
        n_epochs = None

    else:
        raise Exception("Invalid modelName!")

    # Print some model info
    print("input_shape_model = {}".format(input_shape_model))
    print("output_shape_model = {}".format(output_shape_model))

    # Apply transformations to all images
    class ImageTransformer(object):

        def __init__(self, shape_resize):
            self.shape_resize = shape_resize

        def __call__(self, img):
            img_transformed = resize_img(img, self.shape_resize)
            img_transformed = normalize_img(img_transformed)
            return img_transformed

    transformer = ImageTransformer(shape_img_resize)
    print("Applying image transformer to training images...")
    imgs_train_transformed = apply_transformer(imgs_train, transformer, parallel=parallel)
    print("Applying image transformer to test images...")
    imgs_test_transformed = apply_transformer(imgs_test, transformer, parallel=parallel)

    # Convert images to numpy array
    X_train = np.array(imgs_train_transformed).reshape((-1,) + input_shape_model)
    X_test = np.array(imgs_test_transformed).reshape((-1,) + input_shape_model)
    print(" -> X_train.shape = {}".format(X_train.shape))
    print(" -> X_test.shape = {}".format(X_test.shape))


    # Create embeddings using pre-trained model 
    print("Inferencing embeddings using pre-trained model...")
    E_train = model.predict(X_train)
    E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))
    E_test = model.predict(X_test)
    E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model)))
    print(" -> E_train.shape = {}".format(E_train.shape))
    print(" -> E_test.shape = {}".format(E_test.shape))
    print(" -> E_train_flatten.shape = {}".format(E_train_flatten.shape))
    print(" -> E_test_flatten.shape = {}".format(E_test_flatten.shape))


    # In order to calculated the nearest neighbours, cosine distance is 
    # calculated between every pair of image. 
    print("Fitting k-nearest-neighbour model on training images...")
    knn = NearestNeighbors(n_neighbors=knn_admin, metric="cosine")
    knn.fit(E_train_flatten)

    # Gives the distance between each images with respective to a single image.
    A = kneighbors_graph(E_train_flatten, knn_admin, mode='distance', include_self=True)
    print('Distance between images:')
    print(A.toarray())


    # Perform image retrieval on test images
    print("Performing image retrieval on test images...")

    # i is the test image index, emb_flatten is the flatten image vector
    for i, emb_flatten in enumerate(E_test_flatten):
        #indices has the index of product_images according to the similarity.
        _, indices = knn.kneighbors([emb_flatten]) # find k nearest train neighbours
        img_query = imgs_test[i] # query image
        imgs_retrieval=[]
        name=[]
        for idx in indices.flatten():
            imgs_retrieval.append(imgs_train[idx])  # retrieval images
            name.append(img_list[idx])
        # converts array to image and saves to the output dir
        for i, img in enumerate(imgs_retrieval):
            plt.figure(figsize=(2.66, 2))
            plt.box(False)
            plt.axis('off')
            plt.imshow(img)
            outFile = os.path.join(outDir, "{}_{}".format(i,name[i])) # saving the retrieved images in output dir.
            plt.savefig(outFile, bbox_inches="tight", pad_inches = 0)
            plt.close()
    

    
    

