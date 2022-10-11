from django.http.request import HttpRequest
from django.http.response import HttpResponse
from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from numpy import array


def salaryML(request):
    if(request.method =='GET'):
        return render(request,'index.html')
    else:
        data = request.POST['value']
        model = pickle.load(open('salary.pkl','rb'))
        ans = model.predict([[data]])
        ans = list([ans])
        if(ans[0]<0 and ans[0]<50): ans[0] = 0
        return render(request,'index.html',{'ans':str(ans[0])})

def result(request):
    salary1 = pd.read_csv("Salary_Data.csv")
    x = salary1['YearsExperience']
    y = salary1['Salary']
    plt.plot(x, y)
    plt.scatter(x, y, color='g')
    y = salary1[['Salary']]  # dependant variable
    x = salary1[['YearsExperience']]  # independent variable
    x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    

    var1 = float(request.GET["value"])
    pred = lr.predict(array([var1]))
    pred = round(pred[0])

    price = "The Predicted Salary is $"+str(pred)

    return render(request, "index.html", {"ans":price})