from django.shortcuts import render
from django.shortcuts import redirect,HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from django.conf import settings
from django.contrib.auth.decorators import login_required

@login_required(login_url='Login')
def index(request):
  return render(request,'Index.html')


def LoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=User.objects.filter(username=username)
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
          login(request,user)
          return redirect('Index')
        else:
          result="Password Entered is wrong"
          return HttpResponse ("Username or Password is incorrect!!!")
  
    return render (request,'Login.html')

def SignupPage(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')

        if pass1!=pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        
        else:
            my_user=User.objects.create_user(uname,email,pass1)
            my_user.save()
            return redirect('Login')
        
    return render (request,'signup.html')

def classify(img_file):
    data = []
    labels = []
    classes = 5
    cur_path = "..\\code\\"#os.getcwd() #To get current directory


    classs = {  0:"Kingfisher",
        1:"Parrot",
        2:"Peacock",
        3:"Pigeon",
        4:"Quetzal" 
}


    
   # Construct the dynamic path to your model
    model_path = os.path.join(settings.BASE_DIR, 'code', 'my_model.h5')

# Load the model
    model = load_model(model_path)
    print("Loaded model from disk");
    path2="uploads//"+img_file
    print(path2)
    test_image = Image.open(path2)
    test_image = test_image.resize((30, 30))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = np.array(test_image)
        #result = model.predict_classes(test_image)[0]	
    predict_x=model.predict(test_image)
    result=np.argmax(predict_x,axis=1)
    sign = classs[int(result) ]        
    print(sign) 
    return sign

@login_required(login_url='Login')
def Classification(request):
    if request.method == 'POST':
        if 'myfile' in request.FILES:
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            filename = fs.save("uploads//"+myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            result=classify(myfile.name)
            return render(request, 'Classification.html', {'uploaded_file_url': uploaded_file_url,'result':"Predicited result  " +result })
        else:
            result = "Please upload an image."
            return render(request, 'Classification.html', {'result':result })
    return render(request,'Classification.html')

def userlogout(request):
    logout(request)
    return redirect('/Login')


    