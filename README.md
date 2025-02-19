#	start python server

1. open ./python in cli

2. run "python server.py"


#	python request to server via curl

1. open project root directory in cli

2. curl -X POST -F "image=@test_image/image_a.jpg" http://127.0.0.1:5000/upload  -o "./temp/response_py.json"

3. responce is logged in "./temp/response_py.json"


#	start frontend server


1. Open ".\frontend" in cli

2. run "python -m http.server 8000"

3. open http://localhost:8000/ in browser


#	useful 3rd party services


1. decode base64 to image 	- https://base64.guru/converter/decode/image

2. pretty view for json 	- https://jsonviewer.stack.hu/
