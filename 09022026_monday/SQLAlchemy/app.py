from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

# Connecting to SQL Database
try:
    engine = create_engine("mssql+pyodbc://localhost/om?driver=ODBC+Driver+17+for+SQL+Server")
    print("Successfully Connected.")
except SQLAlchemyError as e:
    print("Connection Failed",e)

# Define ORM Model
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True,autoincrement=True)
    name = Column(String(50),nullable=False)
    age = Column(Integer,nullable=False)

    def __repr__(self):
        return f"User:- (id={self.id}, name='{self.name}', age={self.age})"

# Base.metadata.create_all(engine) (This line creates the table in the SQL Server Management Studio)

# Create Session
Session = sessionmaker(bind=engine)
session = Session()
print("Session Created")

# CRUD Operations
# Create one record
try:
    new_user = User(name="Om", age = 21)
    session.add(new_user)
    session.commit()
    print("Successfully Created User")
except SQLAlchemyError as e:
    session.rollback()
    print("Insertion Failed",e)

# Adding multiple records at once
try:
    new_users = [
        User(name="Harsh", age = 21),
        User(name="Kamlesh", age = 22),
        User(name="Sam", age = 23),
        User(name="Alok", age = 24),
        User(name="Yash", age = 25),
    ]
    session.add_all(new_users)
    session.commit()
    print("Successfully Created Users")
except SQLAlchemyError as e:
    session.rollback()
    print("Insertion Failed",e)

# Read
try:
    users = session.query(User).all()
    for user in users:
        print(user)
except SQLAlchemyError as e:
    session.rollback()
    print("Read Failed",e)

# Update
try:
    updated_details = session.query(User).filter_by(name="Om").first()
    if updated_details:
        updated_details.age = 22
        session.commit()
        print("Successfully Updated Details")
except SQLAlchemyError as e:
    session.rollback()
    print("UpdateFailed" ,e)

# Delete
try:
    deleted_details = session.query(User).filter_by(name="Om").first()
    if deleted_details:
        session.delete(deleted_details)
        session.commit()
        print("Successfully Deleted Details")
except SQLAlchemyError as e:
    session.rollback()
    print("Delete Failed" ,e)

# Closing session
session.close()










