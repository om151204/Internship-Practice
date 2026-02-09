from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from database import db
from typing import Optional

router= APIRouter()
class Student(BaseModel):
    name: str
    marks: list

class Patch_Student(BaseModel):
    name: Optional[str] = None
    marks: Optional[list] = None


@router.put("/student/{std_id}", tags=["Update a Record"])
async def put_student(std_id:int, data: Student):
    if std_id in db:
        db[std_id] = data
        return f"Student {std_id} updated successfully"
    else:
        raise HTTPException(status_code=404, detail="Student not found")

@router.patch("/student/{std_id}", tags=["Partially Updating a Record"])
async def patch_student(std_id:int, data: Patch_Student):
    if std_id in db:
        if data.name:
            db[std_id]["name"] = data.name
            return f"Student {std_id} updated successfully"
        elif data.marks:
            db[std_id]["marks"] = data.marks
            return f"Student {std_id} updated successfully"
    else:
        raise HTTPException(status_code=404, detail="Student not found")

