import os
import pandas as pd
import webcolors
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from scipy.spatial import KDTree

def get_path():
#getting the current working dir
    root=os.path.join(os.getcwd(), "data", "product_images")
# list all the files from the dir
    image_list= os.listdir(root)
    return root

#filter files with jpg extension
#pictures=[]
#for file in image_list:
 #   if file.endswith(".jpg") or file.endswith(".JPG"):
  #      pictures.append(file)

# names has the color names and positions has the rgb code repestively. 
#names[0]==positions[0] have color name corresponding to the rgb code.
hexnames = webcolors.css3_hex_to_names
names = []
positions = []

for i in hexnames:
    names.append(hexnames[i])
    positions.append(webcolors.hex_to_rgb(i))


def feature_extraction(pictures,names=names,postions=positions):
    print("this is uploaded",pictures)
    root = get_path()
    # creates a kdtree with rgb values.
    spacedb = KDTree(positions)
    img_name=[] #will hold image name
    category=[] #will hold image category
    feature=[] #will hold image color
    for i in pictures:
        args = os.path.join(root, i)
        print('this is dir',args)
        #reads and predicts the category of the image 
        im = cv2.imread(args)
        print(im)
        bbox, label, conf = cv.detect_common_objects(im)
        #bounding box is created.
        output_image = draw_bbox(im, bbox, label, conf)
        image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        print('this is bbox',bbox,label,conf)
        ROI=image
        for box in bbox:
            print(box)
            x,y,w,h = box
            ROI = image[y:h, x:w]
            
        # Read image and print dimensions
        image = ROI

        print("Shape of Image is ",image.shape)
        # appends red, green, blue values in r,g,b lists.
        r,g,b=[],[],[]
        for row in image:
            for temp_r, temp_g, temp_b in row:
                r.append(temp_r)
                g.append(temp_g)
                b.append(temp_b)
        #average of the listed colors
        red= sum(r)/len(r)
        green= sum(g)/len(g)
        blue= sum(b)/len(b)
        
        querycolor = (red,green,blue)
        dist, index = spacedb.query(querycolor)# finds the closest color with closest rgb value
        img_name.append(i)
        category.append(label)
        feature.append(names[index])
        print('The color %r is closest to %s.'%(querycolor, names[index]),label)
    #Creating Dataframe and converts into CSV file.
    #df = pd.DataFrame({'image_name':img_name,'category':category,'feature':feature})
    #df.to_csv('test.csv',sep='|')
    return  category,feature

