🖥 2022 AI Project

Building robots using actual ROS and implementing functions such as NLP, object detection, and tracking under the theme of smart cart carriers in the airport
Implementing algorithms by focusing on NLP among many functions

# 🛩️ On-Airport Project : Autonomous driving service robot in the airport (Smart Cart carrier)

## ⚙️ Main Function 
  - **🔎 Object Detection & Object Tracking**
    - Recognize People/Objects by using HOG(Histogram of Oriented Gradient)
    - Cascade Classifier recognizes whose face it is.
    
  - **💬 NLP**
    - Using Google TTS & STT to recognize customer demand.
    - After the customer's use is over, the robot receives the review by voice and classifies its positives and negatives.
    
  - **🚘 Autonomous Driving**
    - Using SLAM for mapping the area. (assume it is like the airport)
    - When the customer targets the place, the robot navigates itself to the place.
    
## 🛠️ Hardware
  - TurtleBot3
  - Raspberry Pi3 & Raspberry Pi4
  - RaspberryPi-Display
    
## 📄 File Description 
  - B3_AIProject.pdf : PPT data used in the presentation
  - Emotional_Classification(CNN_Classifier).ipynb : Code for developing Emotional Classification using CNN Classifier
  - Emotional_Classification(LSTM).py : Code for developing Emotional Classification using LSTM
  - Google_TTS.py : Use Google TTS in Raspberry Pi 4
  - Pos_Neg_Review_Classifier.ipynb : Get the customer's review by voice and classify the text is whether positive or negative
  - QR_Data_Crawling.ipynb : Save customer's information on the web page and crawl the data by using QR
