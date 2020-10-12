# <h1> CONTENT BASED IMAGE RETRIEVAL </h1>

CREDITS: https://github.com/ankonzoid/artificio/tree/master/image_retrieval

<h2>The project is inspired from the above mentioned link and the medium article https://towardsdatascience.com/find-similar-images-using-autoencoders-315f374029ea</h2>

<h4>
Model.py contains Resnet50 and Vgg19 models used for (Transfer Learning) these models are trained on a large number of images as a result they have weigths generalized for many categories.</h4>
<h4>
●	Pre-trained neural-nets contain weights that are adjusted during the training of the models. ● These weights are stored, so they can be used in other projects.
●	We will download these weights, the Neural Networks extracts the features from images by converting the images in an vector representation. ● We will store the vector representation of the database images, the query image will be converted to a vector using the same network. ● To find the similar images we will calculate the distance between the Query image (vector representation) and the (vector representation) of databasee images. ● In the end KNN algorithm will used to find the k-nearest images to the Query image. 
</h4>

<h3> In order to interact with the model we have create a web apllication using Flask framework. The Query will be select from the home page by clicking on the image.</h3>

![](/images/home.png)
<br>

<h3> From the Admin page we can add new images to the database, and change settings of the model (i.e choose RESNET50 or VGG19) with desired no. of o/p images. </h3>

![](/images/admin.png)
<br>

<h2>  VGG19 OUTPUT </h2>
![](/images/vgg19.png)





app.py is a flask file (used for running the application)

model.py contains the Resnet50 and Vgg19 models

image_by_group uses the csv file to group the names according to the category.

data includes the images used by the model.

src includes image transformation functions and paths.

output has the ouput for Vgg19 and resnet50 models.
