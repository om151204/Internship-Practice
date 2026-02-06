from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()


class Student(BaseModel):
    # id: int
    name: Optional[str] = None
    marks: Optional[list] = None
    # name: str
    # marks: int


db = {1: {"name": "om", "marks": [10, 20, 30]},
      2: {"name": "harsh", "marks": [10, 20, 30]}}


@app.get("/")
def menu():
    return f"Welcome to Student Management System enter /student to add a new student"


@app.post("/student")
def add_student(student_data: Student):
    new_id = max(db.keys()) + 1
    db[new_id] = student_data
    return {f"Student {new_id} added successfully"}


@app.get("/student/fetch_student")
def get_student():
    return db


# @app.get("/student/{student_id}")
# def new_method(student_id):
#     return db[int(student_id)]

@app.put("/student/{std_id}")
def update_student(std_id: int, student_data):
    if std_id in db:
        db[std_id] = student_data
        return {f"Student {std_id} updated successfully"}
    else:
        return {f"Student {std_id} does not exist"}


@app.delete("/student/{std_id}")
def delete_student(std_id: int):
    if std_id in db:
        db.pop(std_id)
        return {f"Student {std_id} deleted successfully"}
    else:
        return {f"Student {std_id} does not exist"}


@app.patch("/student/{std_id}")
def patch_student(std_id: int, student_data: Student):
    if std_id in db:
        if student_data.name:
            db[std_id]["name"] = student_data.name
        else:
            db[std_id]["marks"] = student_data.marks
        return {f"Student {std_id} updated successfully"}
    else:
        return {f"Student {std_id} does not exist"}
