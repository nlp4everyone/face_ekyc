# pip install locust==2.32.5
from locust import HttpUser, task, between
import os
file_path = "images/western_person/validation_faces/rachel_mcadam4.jpeg" # Change to desired path

class FaceRetrievalUser(HttpUser):
    wait_time = between(1, 2)  # Simulate user wait between 1-3 seconds

    @task
    def post_face_retrieve(self):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        with open(file_path, "rb") as image_file:
            files = {
                "file": (file_path, image_file, "image/jpeg")
            }
            data = {
                "similarity_top_k": "3"
            }

            self.client.post(
                "/development/face_retrieve",
                files=files,
                data=data,
                headers={"accept": "application/json"}
            )
