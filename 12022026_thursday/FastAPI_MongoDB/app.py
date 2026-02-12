from pymongo import MongoClient
from fastapi import FastAPI,status,HTTPException
from pydantic import BaseModel, StrictStr,StrictInt, StrictFloat,Field
from typing import List

app = FastAPI()

def connect_mongodb():
    """
    Connect to MongoDB
    :return: Collections
    """
    try:
        client = MongoClient("mongodb://localhost:27017/")
        print("Connected to MongoDB")

        database = client.get_database("Om")
        collection = database["fastAPI_mongoDB_tutorial"]
        return collection

    except ConnectionError as e:
        print("Connection Error: ",e)

class Student(BaseModel):
    name: StrictStr
    age: StrictInt = Field(gt = 0,lt = 30)
    gender: StrictStr
    marks:List[StrictFloat] = List[Field(gt = 0,lt = 100)]

# Create
@app.post("/student",status_code=status.HTTP_201_CREATED)
def add_student(req_data: Student):
    """
    Add student to database
    :param req_data: dict of student data
    :return: Success message along with id and name of student
    """
    collection = connect_mongodb()
    total_students = [i for i in collection.find()]
    _id = len(total_students) + 1
    collection.insert_one({
        "_id": _id,
        "name":req_data.name,
        "age":req_data.age,
        "gender":req_data.gender,
        "marks":req_data.marks,
    })
    return f"Student {req_data.name} added successfully with id {_id}"

# Read
@app.get("/student",status_code=status.HTTP_200_OK)
def get_student():
    """
    Get all students data
    :return: Record of all the students
    """
    collection = connect_mongodb()
    db_records = [i for i in collection.find()]
    return db_records

@app.get("/student/{id_}",status_code=status.HTTP_200_OK)
def get_student(id_: int):
    """
    Get student data from their id
    :param id_: int
    :return: Student data with the provided id
    """
    collection = connect_mongodb()
    db_rec = collection.find_one({"_id": id_},{"_id":0})
    if db_rec:
        return db_rec
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="Student not found")

# Update
@app.put("/student/{id_}",status_code=status.HTTP_202_ACCEPTED)
def put_student(req_data: Student,id_:int):
    """

    :param req_data: Student
    :param id_: int
    :return: Success message along with id of the student
    """
    collection = connect_mongodb()
    db_rec = collection.find_one({"_id":id_})
    if db_rec:
        collection.update_one({"_id": id_},{"$set":req_data.__dict__})
        return f"Student {id_} updated successfully"
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="Student not found" )













