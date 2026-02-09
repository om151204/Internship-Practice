from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from patch_put import router
from database import db
import uvicorn

app = FastAPI()


class Student(BaseModel):
    # id: int
    name: str
    marks: list
    # name: str
    # marks: int

@app.get("/",tags=["Home Page"])
async def menu():
    return f"Welcome to Student Management System enter /student to add a new student"


@app.post("/student", tags=["Adding New Student"])
async def add_student(student_data: Student):
    new_id = max(db.keys()) + 1
    db[new_id] = student_data
    return {f"Student {new_id} added successfully"}


@app.get("/student/fetch_student", tags=["Displaying Records"])
async def get_student():
    return db

# @app.get("/student/{student_id}")
# def new_method(student_id):
#     return db[int(student_id)]

@app.delete("/student/{std_id}", tags=["Deleting a Record"])
async def delete_student(std_id: int):
    if std_id in db:
        db.pop(std_id)
        return {f"Student {std_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Student does not exist")

app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )

