# Install the python requirement first
pip install -r requirements.txt

# after installation of the requirement, install the tensorflow with cuda for RTX support (optional)
pip install 'tensorflow[and-cuda]'

# Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# you can change the face recog model, backbone,
# FaceRecog model
Facenet512, Human-beings, Facenet, Dlib, VGG-Face, ArcFace, GhostFaceNet, SFace, OpenFace, DeepFace, DeepID

# try run the code by using this command
python app.py