# locustfile.py
from locust import HttpUser, task, between
import io

class BackgroundRemovalUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def upload_image(self):
        # Create a dummy image for testing
        image = io.BytesIO()
        image.name = 'test.jpg'
        image.seek(0)

        self.client.post("/remove-background", files={"files": image})

    @task
    def check_task_status(self):
        # Assuming you have a way to get task IDs
        # For simplicity, we're using a dummy task ID here
        self.client.get("/task/dummy_task_id")