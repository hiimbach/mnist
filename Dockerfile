FROM python:3.9

# Set workdir
WORKDIR /usr/src/app

# Copy all files to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt 

# Define port to be used 
EXPOSE 9966

# RUN API 
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
