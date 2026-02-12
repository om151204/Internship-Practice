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

        users = self.collection.find({"age": {"$gt": 21}})
        for user in users:
            print(user)

    def update_records(self):
        prev = {"name":"Harsh Mistry"}
        new1 = {"$set":{"marks":[100,100,100]}}
        self.collection.update_one(prev, new1)
        print("One Collection updated")
        prev = {"name":"Harsh Piyushkumar Mistry"}
        new2 = {"$set":{"marks":[1,1,1]}}
        self.collection.update_many(prev,new2)
        print("Many Collections updated")
        """
        Below method updates the marks list of Om Amit Mishra
        from [100,90,95] to [100,200,95]
        """
        # The filter finds the document AND identifies the '2' inside the array
        filter_query = {"marks": 90}
        # The '$' acts as a placeholder for the index where '2' was found
        update_query = {"$set": {"marks.$": 200}}
        self.collection.update_one(filter_query, update_query)
        print("Updated marks for om from [100,90,95] to [100,200,95]")

        self.collection.update_many({"age":{"$lt":30}}, {"$set":{"category":"interns"}})
        print("Updated records")

        self.collection.update_many({"age":{"$lt":25,"$gt":21}}, {"$set":{"is_active":False}})

    def delete_records(self):
        self.collection.delete_one({"name":"Samarth Prajapati"})
        print("Record deleted")
        self.collection.delete_many({"name":"Harsh Mistry"})
        print("Many Collections deleted")

    def pass_or_fail(self):
        """
        This function does if-else in pymongo.
        :return:
        """
        students_data = self.collection.find()
        for student in students_data:
            total_marks = sum(student.get("marks",[]))
            if total_marks > 200:
                result_status = "Pass"
            else:
                result_status = "Fail"
            self.collection.update_one({"_id":student["_id"]}, {"$set":{"result":result_status}})
        print("Updated records as pass/fail based on aggregate marks")

    def advance_updates(self):
        """
        $inc: Increments a numeric field by a specific amount only works with numeric values
        $push: Appends a new value to an existing array
        $unset: Removes a specific field from the document and deletes the 'category' field we created earlier
        :return: None
        """
        self.collection.update_one({"name":"Om Amit Mishra"},{"$inc":{"age":1}},)
        print("Incremented age by 1")

        self.collection.update_one({"name":"Om Amit Mishra"},{"$push":{"marks":300}},)
        print("Appended 300 in the marks list")

        self.collection.update_many({},{"$unset":{"category":""}})
        print("All the categories are now removed")






if __name__ == "__main__":
    obj1 = MongoDemo()
    # obj1.inserting_records()
    # obj1.display_records()
    # obj1.update_records()
    # obj1.inserting_records()
    # obj1.display_records()
    # obj1.delete_records()
    # obj1.display_records()
    # obj1.pass_or_fail()
    obj1.advance_updates()






