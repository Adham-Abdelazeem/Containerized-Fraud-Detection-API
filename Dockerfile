# 1. Start with a lightweight Python base image 
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your API code and your trained model into the container
# (The first '.' means your computer, the second '.' means the container)
COPY main.py .
COPY model/ ./model/

# 6. Tell Docker which port the container should listen on
EXPOSE 8000

# 7. The command to start the Uvicorn server when the container runs

# (this can't be overridden when you run the container)
ENTRYPOINT ["uvicorn"] 

# (these could be overridden when you run the container)
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000"]
