create database om
use om

CREATE TABLE users (
    id INT IDENTITY(1,1) PRIMARY KEY,   -- Auto-incrementing primary key
    name VARCHAR(50) NOT NULL,          -- Name column with max length 50
    age INT                             -- Age column
);

select * from users;
SELECT * FROM [dbo].[users]