import pandas as pd
from sklearn.preprocessing import StandardScaler #imported for scaling
from sklearn.model_selection import train_test_split #for spliting the data into test and train data
from sklearn.linear_model import LogisticRegression

def get_clean_data():
    #imported the data using pandas
    data = pd.read_csv("data/data.csv") #in python location is considered from root 
    
    #cleaning the data(to remove unnecessary attributes)
    data = data.drop(['Unnamed: 32','id'],axis=1)

    #map mal = 1 nd benign = 0 
    data['diagnosis']=data['diagnosis'].map({'M' : 1,'B' : 0})

    #print(data.head()) printed the first five entries
    return data

#model creation
def create_model(data):

    #predictors and target variable X,Y
    X = data.drop(['diagnosis'],axis=1)
    Y = data['diagnosis']

    #sacling the data because some of the variables are in 1000s,100s and 10s so we need to keep it uniform
    scaler = StandardScaler()
    X = scaler.fit_transform(X) #since dagnosis have 0 and 1s it doesn't need to be scaled

    #split the data into train and test data
    x_train,x_test,y_train,y_test = train_test_split(

        X,Y,test_size=0.2,random_state=42    #test_size=0.2 means that 20% of the data for test set, and the remaining 80% for train set.
                                             #random_state used to randomly split the data into train and test sets. same random split is used every time the code is run.
    )

    #train the model
    model = LogisticRegression()
    model.fit(x_train,y_train) #the fit method is used to train a machine learning model on a given dataset by learning the relationship between the input features and the target variable.

    return model,scaler




def main():
    data = get_clean_data()
    #print(data.head()) , #data.info() to check the attributes

    model,scaler =  create_model(data)

    #

if __name__ == '__main__':
    main()