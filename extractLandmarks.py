import mediapipe as mp
import cv2
import os

# Define the output file path
outputPath = "/Users/skyblu/Documents/Code/fazeModel/landmarksLFW.txt"
# you can find the LFW dataset here: http://vis-www.cs.umass.edu/lfw/
LFWDirPath = "path to lfw dataset"

count = 0
mpFaceMesh = mp.solutions.face_mesh.FaceMesh()

# Open the output file for writing
with open(outputPath, 'w') as f:
    # Loop over all images in the LFW dataset
    for root, dirs, files in os.walk(LFWDirPath):
        for fileName in files:
            try:
                if fileName.endswith('.jpg') or fileName.endswith('.jpeg'):
                    # Load the image
                    image_path = os.path.join(root, fileName)
                    image = cv2.imread(image_path)
                    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Detect the face landmarks in the image
                    results = mpFaceMesh.process(imageRGB)

                    # Iterate through the first face landmarks object (assuming it is not empty)
                    if results.multi_face_landmarks:
                        # Write the face mesh values to the output file
                        for face_landmarks in results.multi_face_landmarks:
                            for i, landmark in enumerate(face_landmarks.landmark):
                                f.write(f'{fileName},{i},{landmark.x},{landmark.y},{landmark.z}\n')
                count+=1
                print(count)
            except:
                print("error")
                continue
            
                

# Close the output file
f.close()

# Release the resources used by the Face Mesh model
mpFaceMesh.close()

