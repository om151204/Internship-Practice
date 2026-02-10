CREATE DATABASE pyodbc_practice
ALTER DATABASE pyodbc_practice MODIFY NAME= practice_1

USE pyodbc_practice

CREATE TABLE ai_interns(id INT IDENTITY(1,1) PRIMARY KEY, name VARCHAR(100) NOT NULL, email VARCHAR(150) UNIQUE NOT NULL);
