from flask import Flask, request, render_template,redirect,url_for,jsonify

app = Flask(__name__)

my_dict = {}

@app.route("/add_student",methods = ["POST"])
def add_student():
    data = request.get_json()
    name = data["name"]
    age = data["age"]
    marks = data["marks"]
    my_dict[name] = {"age":age,"marks":marks}
    return f"Student {name} added successfully"

@app.route("/update_student",methods = ["POST"])
def update_student():
    data = request.get_json()
    name = data["name"]
    age = data["age"]
    marks = data["marks"]
    my_dict[name] = {"age":age,"marks":marks}
    return f"Student {name} updated successfully"

@app.route("/delete_student",methods = ["DELETE"])
def delete_student():
    data = request.get_json()
    name = data["name"]
    my_dict.pop(name)
    return f"Student {name} deleted successfully"

@app.route("/get_student",methods = ["GET"])
def get_student():
    return my_dict


@app.route("/form",methods = ["GET","POST"])
def form():
    if request.method == "GET":
        return render_template("form.html")
    elif request.method == "POST":
        name = request.form["name1"]
        age = request.form["age"]
        marks = request.form["marks"]
        my_dict[name] = {"age":age,"marks":marks}
        return redirect(url_for("get_student"))

@app.route("/json_response",methods = ["GET"])
def json_response():
    return jsonify({"Name": "Om and Harsh", "id": 1, "email": "om@gmail.com harsh@gmail.com"})

if __name__ == "__main__":
    app.run(debug=True)