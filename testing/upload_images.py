import aiohttp
import asyncio
from aiohttp import FormData
import os, time
url = 'http://localhost:8989/development/face_register' # Change url to correct url hosting
folder_path = "../images/western_person/identical_faces"  # Change facial images directory path to correct

async def send_face_register_request(face_id, face_name, file_path):
    headers = {
        'accept': 'application/json'
    }

    # Create form data
    data = FormData()
    data.add_field('face_id', face_id)
    data.add_field('face_name', face_name)
    data.add_field('file',
                   open(file_path, 'rb'),
                   filename=os.path.basename(file_path),
                   content_type='image/jpeg')

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as response:
            return await response.json()


async def main():
    # Generate requests with incremental face_id and face_name based on filename
    requests = []
    for i, file_path in enumerate(os.listdir(folder_path)):
        # Generate face_id (MNV_ followed by padded number)
        face_id = f'MNV_{i}'

        # Generate face_name from filename (without extension)
        face_name = os.path.splitext(os.path.basename(file_path))[0]

        image_path = os.path.join(folder_path,file_path)
        requests.append({
            'face_id': face_id,
            'face_name': face_name,
            'file_path': image_path
        })

    # Send requests concurrently
    tasks = [send_face_register_request(req['face_id'], req['face_name'], req['file_path'])
             for req in requests]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Process responses
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            print(f"Request {i + 1} failed: {response}")
        else:
            print(f"Request {i + 1} response: {response}")


if __name__ == '__main__':
    begin = time.perf_counter()
    asyncio.run(main())
    print(f"End in {round(time.perf_counter() - begin,3)}s")