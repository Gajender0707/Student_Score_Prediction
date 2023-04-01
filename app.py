from flask import Flask,request,render_template
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

app=Flask(__name__)

#load the pickle model
model=pickle.load(open("Model1.pkl","rb"))
Scaler=StandardScaler()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict",methods=["POST","GET"])
def prediction():
    if request.method=="POST":
        gender=request.form["gender"]
        male=0
        female=0
        if gender=="male":
            male=1
            female=0
        elif gender=="female":
            male=0
            female=1

        ethinicity=request.form["ethinicity"]
        group_a=0
        group_b=0
        group_c=0
        group_d=0
        group_e=0
        if ethinicity=="group_a":
            group_a=1
            group_b=0
            group_c=0
            group_d=0
            group_e=0   

        elif ethinicity=="group_b":
            group_a=0
            group_b=1
            group_c=0
            group_d=0
            group_e=0  
        elif ethinicity=="group_c":
            group_a=0
            group_b=0
            group_c=1
            group_d=0
            group_e=0  

        elif ethinicity=="group_d":
            group_a=0
            group_b=0
            group_c=0
            group_d=1
            group_e=0 

        elif ethinicity=="group_e":
            group_a=0
            group_b=0
            group_c=0
            group_d=0
            group_e=1

        parental_education=request.form["parental_education_level"]
        some_collage=0
        associate_degree=0
        high_school=0
        some_high_school=0
        bachelor_degree=0
        master_degree=0

        if parental_education=="some_collage":
                    some_collage=1
                    associate_degree=0
                    high_school=0
                    some_high_school=0
                    bachelor_degree=0
                    master_degree=0

        elif parental_education=="associate_degree":
                    some_collage=0
                    associate_degree=1
                    high_school=0
                    some_high_school=0
                    bachelor_degree=0
                    master_degree=0

        elif parental_education=="high_school":
                    some_collage=0
                    associate_degree=0
                    high_school=1
                    some_high_school=0
                    bachelor_degree=0
                    master_degree=0

        elif parental_education=="some_high_school":
                    some_collage=0
                    associate_degree=0
                    high_school=0
                    some_high_school=1
                    bachelor_degree=0
                    master_degree=0

        elif parental_education=="bachelor_degree":
                    some_collage=0
                    associate_degree=0
                    high_school=0
                    some_high_school=0
                    bachelor_degree=1
                    master_degree=0

        elif parental_education=="master_degree":
                    some_collage=0
                    associate_degree=0
                    high_school=0
                    some_high_school=0
                    bachelor_degree=0
                    master_degree=1
        
        lunch=request.form["lunch"]
        standard=0
        free_reduced=0

        if lunch=="standard":
                       standard=1
                       free_reduced=0

        elif lunch=="free_reduced":
                       standard=0
                       free_reduced=1

        test_preparation=request.form["test_preparation_course"]
        completed=0
        none=0
        if test_preparation=="completed":
                completed=1
                none=0
        elif test_preparation=="none":
                completed=0
                none=1
         

        d={"gender":{"male":male,"female":female},
           "race/ethinicity":{"group_a":group_a,"group_b":group_b,"group_c":group_c,"group_d":group_d,"group_e":group_e},
           "Parental Eduction Level":{"some collage":some_collage,"associates degree":associate_degree,"high school":high_school,
                                      "some high school":some_high_school,"bachelor degree":bachelor_degree,"master degree":master_degree},
            "Lunch":{"standard":standard,"free/reduced":free_reduced},
            "Test preparation":{"completed":completed,"none":none}

                                      }
                 
        cat_features=[male,female,group_a,group_b,group_c,group_d,group_e,some_collage,associate_degree,high_school,some_high_school,
                               bachelor_degree,master_degree,standard,free_reduced,completed,none]


        output=model.predict([cat_features])
        pred=round(output[0],0)

        return render_template("home.html",pred=pred)


if __name__=="__main__":
    app.run(debug=True)

