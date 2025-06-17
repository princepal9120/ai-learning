# USE PYTHON 3.11 BASE IMAGE

FROM python:3.11-slim

# set working directory

WORKDIR /app

#copy the requirements file to the working directory

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the content of the local src directory to the working directory

COPY . .

# expose port 8000  

EXPOSE 8000

# command to run on container start

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]  