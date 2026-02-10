import pyodbc

print(pyodbc.drivers())

conn = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=practice_1;"
    "Trusted_Connection=yes;"
)

try:
    conn = pyodbc.connect(conn)
    print("Connection Successful.")
except ValueError:
    print("Connection Error.")
except Exception as e:
    print("Connection Error.")


cursor = conn.cursor()

#Performing CRUD Operations
#Inserting records
# records = [("Om","om@gmail.com"),("samarth","samarth@gmail.com"),("kamlesh","kamlesh@gmail.com")]
# cursor.executemany("INSERT INTO ai_interns(name,email) VALUES (?,?)",records)
# print("Executed Successfully")
# conn.commit()

# Retrieve Records
cursor.execute("SELECT * FROM ai_interns")
datas = cursor.fetchall()
for data in datas:
    print(f"Id: {data[0]}, Name:{data[1]}, email:{data[2]}")

# # Update Record
# query = "UPDATE ai_interns SET name=? WHERE name=?"
# cursor.execute(query, ("Om Mishra","Om"))
# print("Record Updated Successfully")
# conn.commit()

# # Delete Record
# delete_query = "DELETE FROM ai_interns WHERE name=?"
# cursor.execute(delete_query, ("Kamlesh",))
# conn.commit()

cursor.close()
conn.close()