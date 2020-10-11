import pandas as pd
import os

#get working dir
root=os.path.join(os.getcwd(), "catlog_categories.csv")
# read the csv file from the folder
df = pd.read_csv(root,index_col=0)

def check_id(x):
    id_unique = df['client_id'].unique()
    if x in id_unique:
        return True
    else:
        return False
# retrives prices of the image
def get_price(x):
    df = pd.read_csv(root,index_col=0)
    lis=[df.loc[df['picture_name'] == i, 'price'].item() for i in x]
    
    return lis
# gets product name
def get_productname(x):
    
    df = pd.read_csv(root,index_col=0)
    lis=[df.loc[df['picture_name'] == i, 'product_name'].item() for i in x]
    
    return lis
# lists unique categories
def unique_categories():
    df = pd.read_csv(root,index_col=0)
    unique_cat = df['category'].unique()
    print(unique_cat)
    return unique_cat
# lists images according to category
def category_admin(cat_admin):
    df = pd.read_csv(root,index_col=0)
    df2 = df.groupby('category')
    df2 = df2.get_group(cat_admin)
    df2 = df2['picture_name'].to_list()
    return df2

# function to retrieve the image names according to the category
def image_group(x):
    df = pd.read_csv(root,index_col=0)
    unique_categories = df['category'].unique()
    
    # get the category of the image
    cat = df.loc[df['picture_name'] == x, 'category'].item()
    
    # group the images according to the category
    df1 = df.groupby('category')
    
    # get the images according to the category in variable cat
    df1 = df1.get_group(cat)
    # converting the dataframe to list.
    df1 = df1['picture_name'].tolist()
    

    return df1
