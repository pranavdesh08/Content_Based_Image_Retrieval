import os
import shutil
import random
from time import localtime, strftime
from uuid import uuid4
import pandas as pd
from datetime import timedelta
from flask import Flask, render_template, request, url_for, send_from_directory, redirect, send_file, session 
from image_by_group import image_group, unique_categories, category_admin, check_id, get_price, get_productname
from model import main_model
from Feature_extraction import feature_extraction
from jinja2 import Environment
from dbmodel.sqlquery import sql_query
from dbmodel.sqlquery import sql_delete, sql_query
from dbmodel.sqlquery import sql_query, sql_query2
from dbmodel.sqlquery import sql_edit_insert, sql_query
import csv

# changing default static folder to catlog_images
app = Flask(__name__, static_folder= 'data/product_images')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

app.secret_key = 'some secret key'

app.config['SESSION_TYPE'] = 'filesystem'
env = Environment(lstrip_blocks=True)
app.jinja_env.filters['zip'] = zip

#default model setting
knn_admin = '5'
model_name  = 'resnet50'


# gets the path of the working directory 
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)

@app.route("/index/<filename>")
def index(filename):
    img_query = [filename] #query image
    session["images"] = filename #add image name to session
    session["end_time"] = strftime("%Y-%m-%d %H:%M:%S", localtime()) #session end time

    img_list = image_group(filename) #groups by category
    
    to_delete = os.path.join(APP_ROOT, 'output/{name}/'.format(name = model_name))
    #deletes existing images
    if os.path.exists(to_delete):
        shutil.rmtree(to_delete)

    # runs the model based on query image, grouped image and no. images to retrive
    main_model(img_list,img_query,model_name,int(knn_admin))

    #to display images on the /index/filename page
    try:
        image_list= category_admin(cat_admin)
    except (NameError,KeyError):
        image_list= os.listdir(os.path.join(APP_ROOT, 'data/product_images'))

    image_output= os.listdir(to_delete)

    random.shuffle(image_list)
    txt = "Similar images"
    user_id = session["USERNAME"]
    start_time = session["start_time"]
    images = session["images"]
    end_time = session["end_time"]
    logs = {'user_id':[user_id], 'start_time':[start_time],'images':[images], 'end_time':[end_time]}
    user_logs = pd.DataFrame(logs)
    user_logs.to_csv('user_tracking.csv', mode='a', header=False) #adding logs to csv file
    print(logs)

    product_name = get_productname(image_list)
    price = get_price(image_list)
    a=zip(image_list[:9],product_name,price)
    print(a)

    strip_output_img= [i[2:] for i in image_output]
    output_product_name = get_productname(strip_output_img)
    output_price = get_price(strip_output_img)

    return render_template ("display.html", filename=zip(image_list[:9],product_name,price), filename1=zip(image_output,output_product_name,output_price), 
                            filename2=txt,knn_admin=int(knn_admin))


# displays the retrieved images   
@app.route("/upload/<filename>")
def send_image(filename):
    to_delete = os.path.join(APP_ROOT, 'output/{name}/'.format(name = model_name))
    return send_from_directory(to_delete,filename)

# displays the image on home page. [Main route]
@app.route('/')
def display_image():
    session.pop("USERNAME", None)
    session.pop("images", None)
    try:
        image_list= category_admin(cat_admin)
        
    except (NameError,KeyError):
        image_list= os.listdir(os.path.join(APP_ROOT, 'data/product_images'))
        
    
    
    secret=str(uuid4())
    session["USERNAME"] = secret
    session["start_time"] = strftime("%Y-%m-%d %H:%M:%S", localtime())
    print(session)

    print('this is image list:',image_list)
    random.shuffle(image_list)
    product_name = get_productname(image_list)
    price = get_price(image_list)

    return render_template('display.html', filename=zip(image_list[:9],product_name, price))

@app.route('/viewcsv') #to view catlog file
def viewcsv():
    
    return redirect(url_for('sql_database'))

@app.route('/user_track') #to view log file
def view_user_track():
    df = pd.read_csv('user_tracking.csv',index_col=0)
    df = df.to_html()
    
    return render_template('view_usertrack.html', table_html = df)

@app.route('/download') # to download catlog file 
def download():
    file= 'catlog_categories.csv'
    return send_file(file,as_attachment=True)

@app.route('/download_user_track') # to download log file 
def user_track():
    file= 'user_tracking.csv'
    return send_file(file,as_attachment=True)

@app.route('/admin', methods=["GET","POST"])
def admin():
    # gets values submitted in the model settings form
    if request.method == 'GET':

        global cat_admin 
        cat_admin=request.args.get('category')
        global knn_admin
        knn_admin=request.args.get('k_nn')
        if knn_admin is None:
            knn_admin = '5'
        global model_name
        model_name=request.args.get('model')
        if model_name is None:
            model_name = 'resnet50'
        print(model_name)

    model = ['vgg19','resnet50']
    target = os.path.join(APP_ROOT, 'data/product_images/')
    target_test = os.path.join(APP_ROOT, 'data/test/')

    # stored the images uploaded 
    if request.method == 'POST':
        file_list = request.files.getlist("file")
        pict_name =[]
        for file in file_list:
            filename = file.filename
            pict_name.append(filename)
            destination = "/".join([target, filename])
            destination_test = "/".join([target_test, filename])
            file.save(destination)
            file.save(destination_test)

        upload_cat=request.form['upload_cat']
        upload_id=request.form['upload_id']
        upload_id=int(upload_id)
        price = request.form['price']
        
        price = price.split(',')
        while len(price) < len(file_list):
            price.append('None')
        upload_id = [upload_id]*len(file_list)
        upload_cat = [upload_cat] * len(file_list)

        product_name,features = feature_extraction(pict_name)
        #saves values in a csv file (catlog file)
        csv_dictionary = {'picture_name':pict_name,'category':upload_cat,'client_id':upload_id,'price':price,
                            'product_name':product_name ,'color':features}
        print(csv_dictionary)

        df_append = pd.DataFrame(csv_dictionary)
        df_append.to_csv('catlog_categories.csv', mode='a', header=False)
        # adding new values to the database
        for i in range(len(pict_name)):
            print(type(pict_name[i]))
            print(type(product_name[i]))
            sql_edit_insert(''' INSERT INTO data_table (`index`,picture_name,category,client_id,price,product_name,color) VALUES (?,?,?,?,?,?,?) ''', (i,pict_name[i],upload_cat[i],upload_id[i],price[i],product_name[i][0],features[i]))
        return redirect(request.path,code=302)

    return render_template('admin.html', categories=unique_categories(),model=model,result={})

@app.route('/sql_database') 
def sql_database():
    # view all the columns and row from the table
    results = sql_query(''' SELECT * FROM data_table''')
    
    return render_template('viewcsv.html', results=results)

@app.route('/delete',methods = ['POST', 'GET']) #this is when user clicks delete link
def sql_datadelete():
    # delete row from the table
    if request.method == 'GET':
        lname = request.args.get('lname')
        fname = request.args.get('fname')
        sql_delete(''' DELETE FROM data_table WHERE picture_name = ? and category = ?''', (fname,lname) )
    results = sql_query(''' SELECT * FROM data_table''')
    # update csv file
    with open('catlog_categories.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['index','picture_name', 'category', 'client_id', 'price', 'product_name', 'color'])
        writer.writerows(results)
    
    return render_template('viewcsv.html', results=results)

@app.route('/query_edit',methods = ['POST', 'GET']) #this is when user clicks edit link
def sql_editlink():
    # retrive values of the row
    if request.method == 'GET':
        elname = request.args.get('elname')
        efname = request.args.get('efname')
        eresults = sql_query2(''' SELECT * FROM data_table WHERE picture_name = ? and category = ?''', (efname,elname))
    results = sql_query(''' SELECT * FROM data_table''')
    return render_template('viewcsv.html', eresults=eresults, results=results)

@app.route('/edit',methods = ['POST', 'GET']) #this is when user submits an edited row
def sql_dataedit():
    # update row values
    if request.method == 'POST':
        old_picture_name = request.form['old_picture_name']
        old_category = request.form['old_category']
        picture_name = request.form['picture_name']
        category = request.form['category']
        client_id = request.form['client_id']
        price = request.form['price']
        product_name = request.form['product_name']
        color = request.form['color']
        sql_edit_insert(''' UPDATE data_table SET picture_name=?,category=?,client_id=?,price=?,product_name=?,color=? WHERE picture_name=? and category=? ''', (picture_name,category,client_id,price,product_name,color,old_picture_name,old_category) )
    results = sql_query(''' SELECT * FROM data_table''')

    with open('catlog_categories.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['index','picture_name', 'category', 'client_id', 'price', 'product_name', 'color'])
        writer.writerows(results)
    
    return render_template('viewcsv.html', results=results)



if __name__=="__main__":
    app.run(debug=True)