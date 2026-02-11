import pymongo

class MongoDemo:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient("mongodb://localhost:27017")
            print("Connected to MongoDB")
        except ConnectionRefusedError:
            print("Could not connect to MongoDB")
        try:
            self.db = self.client["Om"]
            print("Database created")
        except ConnectionRefusedError:
            print("Could not create database")
        try:
            self.collection = self.db["Students"]
            print("Collection created")
        except RuntimeError:
            print("Could not create collection")

    def inserting_records(self):
        data1 = {"name":"Om Amit Mishra","age":21,"marks":[100,90,95]}
        self.collection.insert_one(data1)
        print("Inserted single record")
        data2 = [
            {"name":"Harsh Mistry","age":22,"marks":[12,13,14]},
            {"name":"Kamlesh Patel","age":23,"marks":[11,13,14]},
            {"name":"Samarth Prajapati","age":24,"marks":[12,13,14]},
            {"name":"ALok Yadav","age":25,"marks":[10,13,14]},
        ]
        self.collection.insert_many(data2)
        print(f"Inserted many records")

    def display_records(self):
        database_names = self.client.list_database_names()
        print(database_names)
        col = self.client["Om"]
        print(col.list_collection_names())
        one_record = self.collection.find_one({"name":"Om Amit Mishra"})
        print(one_record)
        multiple_records = self.collection.find({"name":"Kamlesh Patel"})
        for record in multiple_records:
            print(record)

    def update_records(self):
        prev = {"name":"Harsh Mistry"}
        new1 = {"$set":{"marks":[100,100,100]}}
        self.collection.update_one(prev, new1)
        print("One Collection updated")
        prev = {"name":"Harsh Mistry"}
        new2 = {"$set":{"marks":[1,1,1]}}
        self.collection.update_many(prev,new2)
        print("Many Collections updated")

    def delete_records(self):
        self.collection.delete_one({"name":"Samarth Prajapati"})
        print("Record deleted")
        self.collection.delete_many({"name":"Harsh Mistry"})
        print("Many Collections deleted")




if __name__ == "__main__":
    obj1 = MongoDemo()
    obj1.inserting_records()
    obj1.display_records()
    obj1.update_records()
    obj1.inserting_records()
    obj1.display_records()
    obj1.delete_records()
    obj1.display_records()




